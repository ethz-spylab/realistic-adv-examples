import torch
from foolbox.distances import LpDistance, l2

from src.attacks.base import Bounds, DirectionAttack
from src.attacks.queries_counter import AttackPhase, QueriesCounter
from src.model_wrappers import ModelWrapper


class OPTAttackPhase(AttackPhase):
    direction_search = "direction_search"
    direction_probing = "direction_probing"
    gradient_estimation = "gradient_estimation"
    step_size_search = "step_size_search"
    binary_search = "binary_search"


class OPT(DirectionAttack):
    verbose = True

    def __call__(self,
                 model: ModelWrapper,
                 x: torch.Tensor,
                 label: torch.Tensor,
                 target: torch.Tensor | None = None,
                 query_limit: int = 10_000) -> tuple[torch.Tensor, QueriesCounter, float, bool, dict[str, float | int]]:
        if target is not None:
            raise NotImplementedError('Targeted attack is not implemented for OPT')
        return self.attack_untargeted(model, x, label, query_limit)

    def __init__(self,
                 epsilon: float | None,
                 distance: LpDistance,
                 bounds: Bounds,
                 discrete: bool,
                 max_iter: int,
                 alpha: float,
                 beta: float,
                 substract_steps: int = 0):
        super().__init__(epsilon, distance, bounds, discrete)
        self.num_directions = 100 if distance == l2 else 500
        # TODO: we may need a better way for controlling number of queries
        # Can't exactly set max query for line search / binary search
        self.iterations = max_iter
        self.alpha = alpha  # 0.2
        self.beta = beta  # 0.001

    def attack_untargeted(self, model: ModelWrapper, x: torch.Tensor, y: torch.Tensor, query_limit: int | None) -> \
            tuple[torch.Tensor, QueriesCounter, float, bool, dict[str, float | int]]:
        """Attack the original image and return adversarial example
        model: (pytorch model)
        train_dataset: set of training data
        (x0, y0): original image
        """
        queries_counter = QueriesCounter(query_limit)
        alpha, beta = self.alpha, self.beta

        best_theta, prev_best_theta, g_theta = None, None, float("inf")
        print(f"Searching for the initial direction on {self.num_directions} random directions")

        for i in range(self.num_directions):
            theta = torch.randn_like(x)
            success, queries_counter = self.is_correct_boundary_side(model, x + theta, y, None, queries_counter,
                                                                     OPTAttackPhase.direction_search)
            if success.item():
                theta, initial_lbd = normalize(theta)
                lbd, queries_counter = self.fine_grained_binary_search(model, x, y, theta, initial_lbd.item(), g_theta,
                                                                       queries_counter)
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    self.log(f"---> Found distortion {g_theta:.4f}")

        if g_theta == float("inf"):
            # TODO: is this just trying the exact same again?
            best_theta, g_theta = None, float("inf")
            self.log(f"Searching for the initial direction on {self.num_directions} random directions")

            for i in range(self.num_directions):
                theta = torch.randn_like(x)
                success, queries_counter = self.is_correct_boundary_side(model, x + theta, y, None, queries_counter,
                                                                         OPTAttackPhase.direction_search)
                if success.item():
                    theta, initial_lbd = normalize(theta)
                    lbd, queries_counter = self.fine_grained_binary_search(model, x, y, theta, initial_lbd, g_theta,
                                                                           queries_counter)
                    if lbd < g_theta:
                        best_theta, g_theta = theta, lbd
                        self.log(f"---> Found distortion {g_theta:.4f}")

        if g_theta == float("inf"):
            self.log("Couldn't find valid initial, failed")
            return x, queries_counter, float("inf"), False, {}

        self.log(f"====> Found best distortion {g_theta:.4f} using {queries_counter.total_queries} "
                 f"queries and {queries_counter.total_unsafe_queries} extra queries")

        g1 = 1.0
        assert best_theta is not None
        theta, g2 = best_theta, g_theta
        for i in range(self.iterations):
            q = 10
            min_g1 = float("inf")
            gradient = torch.zeros_like(theta)
            u = torch.randn((q, ) + theta.shape, device=theta.device)
            u, _ = normalize(u, batch=True)
            ttt = theta.unsqueeze(0) + beta * u
            ttt, _ = normalize(ttt, batch=True)
            for j in range(q):
                g1, queries_counter = self.fine_grained_binary_search_local(model,
                                                                            x,
                                                                            y,
                                                                            ttt[j],
                                                                            queries_counter,
                                                                            OPTAttackPhase.gradient_estimation,
                                                                            initial_lbd=g2,
                                                                            tol=beta / 500)
                gradient += (g1 - g2) / beta * u[j]
                if g1 < min_g1:
                    min_g1 = g1
                    min_ttt = ttt[j]
            gradient = 1.0 / q * gradient

            if (i + 1) % 10 == 0:
                dist = (g2 * theta).norm().item()
                self.log((f"Iteration {i + 1:3d} distortion {dist:.4f} num_queries {queries_counter.total_queries}, "
                          f"unsafe queries: {queries_counter.total_unsafe_queries}"))

            min_theta = theta
            min_g2 = g2

            for _ in range(15):
                new_theta = theta - alpha * gradient
                new_theta, _ = normalize(new_theta)
                new_g2, queries_counter = self.fine_grained_binary_search_local(model,
                                                                                x,
                                                                                y,
                                                                                new_theta,
                                                                                queries_counter,
                                                                                OPTAttackPhase.step_size_search,
                                                                                initial_lbd=min_g2,
                                                                                tol=beta / 500)
                alpha *= 2
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                else:
                    break

            if min_g2 >= g2:
                for _ in range(15):
                    alpha *= 0.25
                    new_theta = theta - alpha * gradient
                    new_theta, _ = normalize(new_theta)
                    new_g2, queries_counter = self.fine_grained_binary_search_local(model,
                                                                                    x,
                                                                                    y,
                                                                                    new_theta,
                                                                                    queries_counter,
                                                                                    OPTAttackPhase.step_size_search,
                                                                                    initial_lbd=min_g2,
                                                                                    tol=beta / 500)
                    if new_g2 < g2:
                        min_theta = new_theta
                        min_g2 = new_g2
                        break

            if min_g2 <= min_g1:
                theta, g2 = min_theta, min_g2
            else:
                theta, g2 = min_ttt, min_g1  # type: ignore

            if g2 < g_theta:
                best_theta, g_theta = theta, g2

            if alpha < 1e-4:
                alpha = 1.0
                self.log(f"Warning: not moving, g2 {g2:.4f} gtheta {g_theta:.4f}")
                beta *= 0.1
                if beta < 1e-8:
                    break

            # prev_best_theta is kept to make sure that we use the latest theta
            # before max query is reached
            if queries_counter.is_out_of_queries():
                print("Out of queries")
                break

            prev_best_theta = best_theta.clone()

        assert prev_best_theta is not None
        target = model.predict_label(self.get_x_adv(x, prev_best_theta, g_theta)).item()

        self.log(f"\nAdversarial example found: distortion {g_theta:.4f} predicted class {target} queries "
                 f"{queries_counter.total_queries}, unsafe queries: {queries_counter.total_unsafe_queries}")

        distance: float = (x + g_theta * prev_best_theta - x).norm().item()

        return self.get_x_adv(x, prev_best_theta,
                              g_theta), queries_counter, distance, not queries_counter.is_out_of_queries(), {}

    def log(self, arg):
        if self.verbose:
            print(arg)

    def fine_grained_binary_search_local(self,
                                         model: ModelWrapper,
                                         x: torch.Tensor,
                                         y: torch.Tensor,
                                         theta: torch.Tensor,
                                         queries_counter: QueriesCounter,
                                         phase: OPTAttackPhase,
                                         initial_lbd: float = 1.0,
                                         tol: float = 1e-5) -> tuple[float, QueriesCounter]:
        lbd = initial_lbd

        def is_correct_boundary_side_local(lbd_: float, qc: QueriesCounter) -> tuple[torch.Tensor, QueriesCounter]:
            x_adv_ = self.get_x_adv(x, theta, lbd_)
            return self.is_correct_boundary_side(model, x_adv_, y, None, qc, phase)

        x_adv = self.get_x_adv(x, theta, lbd)c
        success, queries_counter = self.is_correct_boundary_side(model, x_adv, y, None, queries_counter,
                                                                 phase)

        if not success:
            lbd_lo = lbd
            lbd_hi = lbd * 1.01
            while not (iter_result := is_correct_boundary_side_local(lbd_hi, queries_counter))[0].item():
                _, queries_counter = iter_result
                lbd_hi *= 1.01
                if lbd_hi > 20:
                    return float('inf'), queries_counter
        else:
            lbd_hi = lbd
            lbd_lo = lbd * 0.99
            while (iter_result := is_correct_boundary_side_local(lbd_lo, queries_counter))[0].item():
                _, queries_counter = iter_result
                lbd_lo *= 0.99

        # EDIT: fix bug that makes Sign-OPT stuck in while loop
        # while lbd_hi - lbd_lo > tol:
        diff = lbd_hi - lbd_lo
        while diff > tol:
            lbd_mid = (lbd_lo + lbd_hi) / 2
            # EDIT: add a break condition
            if lbd_mid == lbd_hi or lbd_mid == lbd_lo:
                break
            success, queries_counter = is_correct_boundary_side_local(lbd_mid, queries_counter)
            if success.item():
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
            # EDIT: This is to avoid numerical issue with gpu tensor when diff is small
            if diff <= lbd_hi - lbd_lo:
                break
            diff = lbd_hi - lbd_lo
        return lbd_hi, queries_counter

    def fine_grained_binary_search(self, model: ModelWrapper, x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor,
                                   initial_lbd: float, current_best: float,
                                   queries_counter: QueriesCounter) -> tuple[float, QueriesCounter]:
        if initial_lbd > current_best:
            x_adv = self.get_x_adv(x, theta, current_best)
            success, queries_counter = self.is_correct_boundary_side(model, x_adv, y, None, queries_counter,
                                                                     OPTAttackPhase.direction_probing)
            if not success.item():
                return float('inf'), queries_counter
            lbd = current_best
        else:
            lbd = initial_lbd

        lbd_hi = lbd
        lbd_lo = 0.0

        # EDIT: This tol check has a numerical issue and may never quit (1e-5)
        while lbd_hi - lbd_lo > 1e-5:
            lbd_mid = (lbd_lo + lbd_hi) / 2
            # EDIT: add a break condition
            if lbd_mid == lbd_hi or lbd_mid == lbd_lo:
                break
            x_adv = self.get_x_adv(x, theta, lbd_mid)
            success, queries_counter = self.is_correct_boundary_side(model, x_adv, y, None, queries_counter,
                                                                     OPTAttackPhase.binary_search)
            if success.item():
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid

        return lbd_hi, queries_counter


def normalize(x: torch.Tensor, batch: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize x in-place
    Args:
        x (torch.Tensor): Tensor to normalize
        batch (bool): First dimension of x is batch
    """
    if batch:
        norm = x.reshape(x.size(0), -1).norm(2, 1)  # type: ignore
    else:
        norm = x.norm()
    for _ in range(x.ndim - 1):
        norm.unsqueeze_(-1)
    x /= (norm + 1e-9)
    if batch:
        norm = norm.view(x.size(0), 1)
    else:
        norm.squeeze_()
    return x, norm
