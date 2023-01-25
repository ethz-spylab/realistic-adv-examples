# Evading Black-box Defenses Without Breaking Eggs

Code to reproduce results of the submission *"Evading Black-box Defenses Without Breaking Eggs"*.

## Environment

The code was developed using Python 3.10.6 and Conda. The environment can be reproduced by creating the Conda environment in [`environment.yml`](./environment.yml) via

```sh
conda env create --file=environment.yml
```

This should install all the libraries needed to run the code. Please note that there could be some issues due to the CUDA version, so make sure that your hardware software stack supports the CUDA versions specified in `environment.yml`.

## Running the experiments

The entry point to run the attacks is the main [`main.py`](main.py) file. In particular, the file supports several arguments to run different attacks with different parameters on different datasets.

### ImageNet $\ell_2$ (Fig. 3.a)

The experiments on ImageNet $\ell_2$ in Figure 3.a can be reproduced with the following commands:

```sh
python main.py --early 0 --dataset resnet_imagenet --norm l2 --num 500 --max-unsafe-queries 15000 --max-queries 100000 --out-dir results $ATTACK_CONFIG
```

where `$ATTACK_CONFIG` corresponds to different parameters for different attacks:

- Boundary: `--attack boundary`
- OPT: `--attack opt --opt-beta 0.01`
- Stealthy OPT: `--attack opt --search line --opt-beta 0.01 --opt-n-searches 2`
- SignOPT: `--attack sign_opt --opt-beta 0.01`
- Stealthy SignOPT: `--attack sign_opt --search line --opt-beta 0.01 --opt-n-searches 2`
- HSJA: `--attack hsja --hsja-gamma 10000`

### ImageNet $\ell_\infty$ (Fig 3.d)

The experiments on ImageNet $\ell_\infty$ in Figure 3.d can be reproduced with the following commands:

```sh
python main.py --early 0 --dataset resnet_imagenet --norm linf --num 500 --max-unsafe-queries 15000 --max-queries 100000 --out-dir results $ATTACK_CONFIG
```

where `$ATTACK_CONFIG` corresponds to different parameters for different attacks:

- HSJA: `--attack hsja --hsja-gamma 10000`
- RayS: `--attack rays`
- Stealthy RayS: `--attack rays --search line --line-search-tol 0.1`

Please note that the attacks generate quite large logs (up to 13GB per experiment), so make sure to have enough space to store the results.

## Plotting

The file [`plot_dist_vs_queries.py`](plot_dist_vs_queries.py) can be used to plot the results generated from the attacks. In particular, after running the commands above, Fig. 3.a can be plotted with the following command:

```sh
python plot_dist_vs_queries.py distance --exp-paths $BOUNDARY_RESULTS_PATH $OPT_RESULTS_PATH $STEALTHY_OPT_RESULTS_PATH $SIGNOPT_RESULTS_PATH $STEALTHY_SIGNOPT_RESULTS_PATH $HSJA_RESULTS_PATH --names Boundary OPT "Stealthy OPT" SignOPT "Stealthy SignOPT" "HSJA" --to-simulate 2 4 --unsafe-only --max-queries 1000 --max-samples 500 --out-path plots/imagenet_l2.pdf
```

And Fig. 3.d can be plotted with the following command:

```sh
python plot_dist_vs_queries.py distance --exp-paths $HSJA_RESULTS_PATH $RAYS_RESULTS_PATH $STEALTHY_RAYS_RESULTS_PATH --names HSJA RayS "Stealthy RayS" --unsafe-only --max-queries 1000 --max-samples 500 --out-path plots/imagenet_linf.pdf
```

## Credits

The attack implementations have been adapted from the official implementations for [OPT](https://github.com/LeMinhThong/blackbox-attack), [SignOPT](https://github.com/cmhcbb/attackbox), [HSJA](https://github.com/Jianbo-Lab/HSJA/), and the Boundary Attack (released as part of [Foolbox](https://github.com/bethgelab/foolbox)). Some parts are also borrowed from the code of the paper [Preprocessors Matter! Realistic Decision-BasedAttacks on Machine Learning Systems](https://github.com/google-research/preprocessor-aware-black-box-attack).
