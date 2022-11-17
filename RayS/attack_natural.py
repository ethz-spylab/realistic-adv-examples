from pathlib import Path

import argparse
import json
import subprocess
import uuid

import numpy as np
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights

import dataset
from general_tf_model import GeneralTFModel
from general_torch_model import GeneralTorchModel
from arch import mnist_model, binary_resnet50, cifar_model, clip_laion_nsfw
from RayS_Single import RayS, SafeSideRayS


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def main():
    parser = argparse.ArgumentParser(description='Hard Label Attacks')
    parser.add_argument('--dataset', default='imagenet', type=str, help='Dataset')
    parser.add_argument('--targeted', default='0', type=str, help='targeted or untargeted')
    parser.add_argument('--norm', default='linf', type=str, help='Norm for attack, linf only')
    parser.add_argument('--num', default=1000, type=int, help='Number of samples to be attacked from test dataset.')
    parser.add_argument('--query', default=10000, type=int, help='Maximum queries for the attack')
    parser.add_argument('--batch', default=1, type=int, help='attack batch size.')
    parser.add_argument('--epsilon', default=0.05, type=float, help='attack strength')
    parser.add_argument('--early',
                        default='1',
                        type=str,
                        help='early stopping (stop attack once the adversarial example is found)')
    parser.add_argument('--search', default='binary', type=str, help='Type of search to use, binary or line')
    parser.add_argument('--line-search-tol',
                        default=None,
                        type=float,
                        help='Tolerance for line search w.r.t. previous iteration')
    parser.add_argument(
        '--out-dir',
        default='/local/home/edebenedetti/exp-results/realistic-adv-examples/rays',
        type=str,
    )
    parser.add_argument(
        '--save-img-every',
        default=None,
        type=int,
    )
    parser.add_argument('--flip-squares',
                        default='0',
                        type=str,
                        help='Whether the attack should flip squares and not chunks of a 1-d vector')
    parser.add_argument('--flip-rand-pixels',
                        default='0',
                        type=str,
                        help='Whether the attack should flip random pixels not chunks of a 1-d vector')
    parser.add_argument('--side',
                        default='unsafe',
                        type=str,
                        help='Whether the attack should happen from the safe side of the boundary')
    args = parser.parse_args()

    targeted = True if args.targeted == '1' else False
    early_stopping = False if args.early == '0' else True
    order = 2 if args.norm == 'l2' else np.inf

    if args.flip_squares == '1' and args.flip_rand_pixels == '1':
        raise ValueError("`--flip-squares` cannot be `1` if also `--flip-rand-pixels` is `1`")

    print(args)

    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        out_dir.mkdir()

    if args.dataset == 'mnist':
        model = mnist_model.MNIST().cuda().eval()
        model = torch.nn.DataParallel(model, device_ids=[0])
        model.load_state_dict(torch.load('model/mnist_gpu.pt'))
        test_loader = dataset.load_mnist_test_data(args.batch)
        torch_model = GeneralTorchModel(model, n_class=10, im_mean=None, im_std=None)
    elif args.dataset == 'cifar':
        model = cifar_model.CIFAR10().cuda().eval()
        model = torch.nn.DataParallel(model, device_ids=[0])
        model.load_state_dict(torch.load('model/cifar10_gpu.pt'))
        test_loader = dataset.load_cifar10_test_data(args.batch)
        torch_model = GeneralTorchModel(model, n_class=10, im_mean=None, im_std=None)
    elif args.dataset == 'resnet':
        model = models.__dict__["resnet50"](weights=ResNet50_Weights.IMAGENET1K_V1).cuda().eval()
        model = torch.nn.DataParallel(model, device_ids=[0])
        test_loader = dataset.load_imagenet_test_data(args.batch)
        torch_model = GeneralTorchModel(model,
                                        n_class=1000,
                                        im_mean=[0.485, 0.456, 0.406],
                                        im_std=[0.229, 0.224, 0.225])
    elif args.dataset == 'binary_imagenet':
        model = binary_resnet50.BinaryResNet50.load_from_checkpoint(
            "checkpoints/binary_imagenet.ckpt").model.cuda().eval()
        model = torch.nn.DataParallel(model, device_ids=[0])
        test_loader = dataset.load_binary_imagenet_test_data(args.batch)
        torch_model = GeneralTorchModel(model, n_class=2, im_mean=[0.485, 0.456, 0.406], im_std=[0.229, 0.224, 0.225])
    elif args.dataset == 'imagenet_nsfw':
        model = clip_laion_nsfw.CLIPNSFWDetector("b32", "checkpoints")
        model = torch.nn.DataParallel(model, device_ids=[0])
        test_loader = dataset.load_imagenet_nsfw_test_data(args.batch)
        torch_model = GeneralTFModel(model,
                                     n_class=2,
                                     im_mean=[0.48145466, 0.4578275, 0.40821073],
                                     im_std=[0.26862954, 0.26130258, 0.27577711])
    else:
        print("Invalid dataset")
        exit(1)

    exp_out_dir = out_dir / f"{args.dataset}_{args.norm}_{args.targeted}_{args.early}_{args.search}_{args.epsilon:.3f}_{args.side}_{uuid.uuid4().hex}"
    while exp_out_dir.exists():
        exp_out_dir = out_dir / f"{args.dataset}_{args.norm}_{args.targeted}_{args.early}_{args.search}_{args.epsilon:.3f}_{args.side}_{uuid.uuid4().hex}"
    exp_out_dir.mkdir()

    print(f"Saving results in {exp_out_dir}")
    exp_args = vars(args)
    exp_args['git_hash'] = get_git_revision_hash()
    with open(exp_out_dir / 'args.json', 'w') as f:
        json.dump(exp_args, f, indent=4)

    if args.side == 'unsafe':
        attack = RayS(torch_model,
                      order=order,
                      epsilon=args.epsilon,
                      early_stopping=early_stopping,
                      search=args.search,
                      line_search_tol=args.line_search_tol,
                      flip_squares=args.flip_squares == '1',
                      flip_rand_pixels=args.flip_rand_pixels == '1')
    elif args.side == 'safe':
        attack = SafeSideRayS(torch_model,
                              order=order,
                              epsilon=args.epsilon,
                              early_stopping=early_stopping,
                              search=args.search,
                              line_search_tol=args.line_search_tol,
                              flip_squares=args.flip_squares == '1',
                              flip_rand_pixels=args.flip_rand_pixels == '1')
    else:
        raise ValueError(f"Invalid attack side: {args.side}")

    stop_dists = []
    stop_queries = []
    stop_bad_queries = []
    stop_wasted_queries = []
    asr = []
    early_stoppings = []
    np.random.seed(0)
    seeds = np.random.randint(10000, size=10000)
    count = 0
    miscliassified = 0
    negatives = 0
    for i, (xi, yi) in enumerate(test_loader):
        if torch_model.n_class == 2 and yi.item() == 0:
            negatives += 1
            print("Skipping as item is negative")
            continue

        print(f"Sample {i}, class: {yi.item()}")
        xi, yi = xi.cuda(), yi.cuda()

        if count == args.num:
            break

        if torch_model.predict_label(xi) != yi:
            miscliassified += 1
            print("Skipping as item is misclassified")
            continue

        np.random.seed(seeds[count])

        target = np.random.randint(torch_model.n_class) * torch.ones(yi.shape,
                                                                     dtype=torch.long).cuda() if targeted else None
        while target and torch.sum(target == yi) > 0:
            print('re-generate target label')
            target = np.random.randint(torch_model.n_class) * torch.ones(len(xi), dtype=torch.long).cuda()

        adv, queries, bad_queries, wasted_queries, dist, succ = attack(xi,
                                                                       yi,
                                                                       target=target,
                                                                       seed=seeds[i],
                                                                       query_limit=args.query)

        if args.save_img_every is not None and (i - miscliassified) % args.save_img_every == 0:
            np.save(exp_out_dir / f"{i}_adv.npy", adv[0].cpu().numpy())
            np.save(exp_out_dir / f"{i}.npy", xi[0].cpu().numpy())

        if succ:
            stop_queries.append(queries)
            stop_bad_queries.append(bad_queries)
            stop_wasted_queries.append(wasted_queries)
            early_stoppings.append(attack.n_early_stopping)

            if dist.item() < np.inf:
                stop_dists.append(dist.item())

        elif early_stopping == False:
            if dist.item() < np.inf:
                stop_dists.append(dist.item())

        asr.append(succ.item())

        count += 1

        print(
            "index: {:4d} avg dist: {:.4f} avg queries: {:.4f} median queries: {:.4f} avg bad queries: {:.4f} median bad queries: {:.4f} avg wasted queries: {:.4f} median wasted queries: {:.4f} asr: {:.4f} \n"
            .format(i, np.mean(np.array(stop_dists)), np.mean(np.array(stop_queries)),
                    np.median(np.array(stop_queries)), np.mean(np.array(stop_bad_queries)),
                    np.median(np.array(stop_bad_queries)), np.mean(np.array(stop_wasted_queries)),
                    np.median(np.array(stop_wasted_queries)), np.mean(np.array(asr))))

    results_dict = {
        "miscliassified": miscliassified,
        "distortion": np.mean(np.array(stop_dists)),
        "success_rate": np.mean(np.array(asr)),
        "mean_queries": np.mean(np.array(stop_queries)),
        "median_queries": np.median(np.array(stop_queries)),
        "mean_bad_queries": np.mean(np.array(stop_bad_queries)),
        "median_bad_queries": np.median(np.array(stop_bad_queries)),
        "mean_wasted_queries": np.mean(np.array(stop_wasted_queries)),
        "median_wasted_queries": np.median(np.array(stop_wasted_queries)),
        "mean_early_stoppings": np.mean(np.array(early_stoppings)),
        "median_early_stoppings": np.median(np.array(early_stoppings)),
    }

    with open(exp_out_dir / "results.json", 'w') as f:
        json.dump(results_dict, f, indent=4)

    np.save(exp_out_dir / "distortion.npy", np.array(stop_dists))
    np.save(exp_out_dir / "queries.npy", np.array(stop_queries))
    np.save(exp_out_dir / "bad_queries.npy", np.array(stop_bad_queries))
    np.save(exp_out_dir / "wasted_queries.npy", np.array(stop_wasted_queries))
    np.save(exp_out_dir / "early_stoppings.npy", np.array(early_stoppings))

    print(f"Saved all the results to {exp_out_dir}")


if __name__ == "__main__":
    main()
