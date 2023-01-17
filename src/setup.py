import json
import os
import subprocess
import uuid
from argparse import Namespace
from pathlib import Path

import torch
from dotenv import load_dotenv
from foolbox.distances import l2, linf
from torch.utils import data
from torchvision import models as models
from torchvision.models import ResNet50_Weights

from src import dataset
from src.arch import binary_resnet50, clip_laion_nsfw, edenai_model, google_nsfw_model, resnet50_cifar10
from src.attacks import HSJA, OPT, BoundaryAttack, RayS, SignOPT
from src.attacks.base import BaseAttack, Bounds, SearchMode
from src.model_wrappers import (EdenAIModelWrapper, GoogleNSFWModelWrapper, ModelWrapper, TFModelWrapper,
                                TorchModelWrapper)

API_KEY_NAME = "EDENAI_API_KEY"
load_dotenv()

DEFAULT_BOUNDS = Bounds(0, 1)

DISTANCES = {"linf": linf, "l2": l2}


def setup_model_and_data(args: Namespace, device: torch.device) -> tuple[ModelWrapper, data.DataLoader]:
    if args.dataset == 'resnet_cifar10':
        input_size = 32
        num_classes = 10
        channel = 3
        inner_model = resnet50_cifar10.resnet_v2(input_shape=(input_size, input_size, channel),
                                                 depth=20,
                                                 num_classes=num_classes)
        inner_model.load_weights("checkpoints/resnet_cifar10.hdf5")
        test_loader = dataset.load_cifar10_test_data(args.batch)
        model = TFModelWrapper(inner_model, n_class=10, channels_last=True)
    elif args.dataset == 'resnet_imagenet':
        inner_model = models.__dict__["resnet50"](weights=ResNet50_Weights.IMAGENET1K_V1).to(device).eval()
        inner_model = torch.nn.DataParallel(inner_model, device_ids=[0])
        test_loader = dataset.load_imagenet_test_data(args.batch, args.data_dir)
        model = TorchModelWrapper(inner_model,
                                  n_class=1000,
                                  im_mean=(0.485, 0.456, 0.406),
                                  im_std=(0.229, 0.224, 0.225))
    elif args.dataset == 'binary_imagenet':
        inner_model = binary_resnet50.BinaryResNet50.load_from_checkpoint("checkpoints/binary_imagenet.ckpt").model.to(
            device).eval()
        inner_model = torch.nn.DataParallel(inner_model, device_ids=[0])
        test_loader = dataset.load_binary_imagenet_test_data(args.batch, args.data_dir)
        model = TorchModelWrapper(inner_model, n_class=2, im_mean=(0.485, 0.456, 0.406), im_std=(0.229, 0.224, 0.225))
    elif args.dataset == 'imagenet_nsfw':
        inner_model = clip_laion_nsfw.CLIPNSFWDetector("b32", "checkpoints")
        model = TFModelWrapper(inner_model,
                               n_class=2,
                               im_mean=(0.48145466, 0.4578275, 0.40821073),
                               im_std=(0.26862954, 0.26130258, 0.27577711),
                               take_sigmoid=False)
        test_loader = dataset.load_imagenet_nsfw_test_data(args.batch)
    elif args.dataset == 'google_nsfw':
        api_key = os.environ[API_KEY_NAME]
        inner_model = edenai_model.GoogleNSFWModel(device, api_key)
        model = EdenAIModelWrapper(inner_model, n_class=2, threshold=args.model_threshold).to(device)
        test_loader = dataset.load_imagenet_nsfw_test_data(args.batch,
                                                           Path("nsfw_filters_results/google_racy_five_indices.npy"))
    elif args.dataset == 'api4ai_nsfw':
        api_key = os.environ[API_KEY_NAME]
        inner_model = edenai_model.API4AINSFWModel(device, api_key)
        model = EdenAIModelWrapper(inner_model, n_class=2, threshold=args.model_threshold).to(device)
        test_loader = dataset.load_imagenet_nsfw_test_data(args.batch,
                                                           Path("nsfw_filters_results/api4ai_nsfw_five_indices.npy"))
    elif args.dataset == 'amazon_nsfw':
        api_key = os.environ[API_KEY_NAME]
        inner_model = edenai_model.AmazonNSFWModel(device, api_key)
        model = EdenAIModelWrapper(inner_model, n_class=2, threshold=args.model_threshold).to(device)
        test_loader = dataset.load_imagenet_nsfw_test_data(
            args.batch, Path("nsfw_filters_results/amazon_suggestive_five_indices.npy"))
    elif args.dataset == 'laion_nsfw_mock':
        api_key = os.environ[API_KEY_NAME]
        inner_model = edenai_model.LAIONNSFWModel(device,
                                                  api_key,
                                                  strong_preprocessing=args.strong_preprocessing == '1')
        model = EdenAIModelWrapper(inner_model, n_class=2, threshold=args.model_threshold).to(device)
        test_loader = dataset.load_imagenet_nsfw_test_data(args.batch)
    elif args.dataset == 'google_cloud_nsfw':
        inner_model = google_nsfw_model.GoogleNSFWModel(device)
        model = GoogleNSFWModelWrapper(inner_model, n_class=2, threshold=args.model_threshold).to(device)
        test_loader = dataset.load_imagenet_nsfw_test_data(args.batch,
                                                           Path("nsfw_filters_results/google_racy_five_indices.npy"))
    else:
        raise ValueError("Invalid model")

    model.make_model_eval()

    return model, test_loader


def setup_attack(args: Namespace) -> BaseAttack:
    base_attack_kwargs = {
        "epsilon": args.epsilon,
        "distance": DISTANCES[args.norm],
        "discrete": args.discrete == '1',
        "bounds": Bounds(),
        "queries_limit": args.max_queries,
        "unsafe_queries_limit": args.max_unsafe_queries
    }
    search = SearchMode(args.search)
    opt_grad_estimation_search = (SearchMode(args.opt_grad_est_search)
                                  if args.opt_grad_est_search is not None else search)
    opt_step_size_search = SearchMode(args.opt_step_size_search) if args.opt_step_size_search is not None else search
    opt_kwargs = {
        "max_iter": args.max_iter,
        "alpha": args.opt_alpha,
        "beta": args.opt_beta,
        "search": search,
        "grad_estimation_search": opt_grad_estimation_search,
        "step_size_search": opt_step_size_search,
        "n_searches": args.opt_n_searches,
        "max_search_steps": args.opt_max_search_steps,
        "batch_size": args.opt_bs,
    }
    if args.attack == "rays":
        if args.rays_flip_squares == '1' and args.rays_flip_rand_pixels == '1':
            raise ValueError("`--flip-squares` cannot be `1` if also `--flip-rand-pixels` is `1`")
        attack_kwargs = {
            "early_stopping": args.early == '1',
            "search": search,
            "line_search_tol": args.line_search_tol,
            "flip_squares": args.rays_flip_squares == '1',
            "flip_rand_pixels": args.rays_flip_rand_pixels == '1'
        }
        return RayS(**base_attack_kwargs, **attack_kwargs)
    if args.attack == "hsja":
        attack_kwargs = {
            "num_iterations": args.max_iter,
            "stepsize_search": args.hsja_stepsize_search,
            "max_num_evals": args.hsja_max_num_evals,
            "init_num_evals": args.hsja_init_num_evals,
            "gamma": args.hsja_gamma,
            "fixed_delta": args.hsja_delta,
        }
        return HSJA(**base_attack_kwargs, **attack_kwargs)
    if args.attack == "opt":
        return OPT(**base_attack_kwargs, **opt_kwargs)
    if args.attack == "sign_opt":
        attack_kwargs = {"num_grad_queries": args.sign_opt_num_grad_queries, "momentum": args.sign_opt_momentum}
        return SignOPT(**base_attack_kwargs, **opt_kwargs, **attack_kwargs)
    if args.attack == "boundary":
        return BoundaryAttack(**base_attack_kwargs)
    else:
        raise ValueError(f"Invalid attack: `{args.attack}`")


def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unable to retrieve hash"


def setup_out_dir(args: Namespace) -> Path:
    out_dir = Path(args.out_dir) / args.dataset / args.norm / args.attack
    out_dir.mkdir(parents=True, exist_ok=True)

    def make_exp_out_dir_name() -> Path:
        return out_dir / f"discrete-{args.discrete}_targeted-{args.targeted}_early-{args.early}_{args.search}" \
                         f"_{args.epsilon if args.epsilon is not None else 0:.3f}_{uuid.uuid4().hex}"

    exp_out_dir = make_exp_out_dir_name()
    while exp_out_dir.exists():
        exp_out_dir = make_exp_out_dir_name()
    exp_out_dir.mkdir()
    print(f"Saving results in {exp_out_dir}")
    exp_args = vars(args)
    exp_args['git_hash'] = get_git_revision_hash()
    with open(exp_out_dir / 'args.json', 'w') as f:
        json.dump(exp_args, f, indent=4)
    return exp_out_dir
