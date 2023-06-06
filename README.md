# Evading Black-box Classifiers Without Breaking Eggs

*Edoardo Debenedetti (ETH Zurich), Nicholas Carlini (Google), Florian Tramèr (ETH Zurich)*

Code to reproduce results of the paper [*"Evading Black-box Classifiers Without Breaking Eggs"*](https://arxiv.org/abs/2306.02895).

## Leaderboard

Leaderboards which show, for each attack, the distances achieved after 100, 200, 500, and 1000 bad queries, and how many total queries have been issued at that amount of bad queries. We sort the attacks by publication date and bold the best result for each column. The leaderboard refers to untargeted attacks against an ImageNet ResNet-50. The results are presented in the form `distance` <sup><sub>(`total_queries`)</sub></sup>. The $\ell_2$ norm is for images in the $[0, 1]$ range, and the $\ell_\infty$ one is for images in the $[0, 255]$ one.

### $\ell_2$

<p align="center">

|                         | 100            | 200            | 500            | 1000           |
|:------------------------|--------------:|--------------:|--------------:|--------------:|
| OPT (2018) | 37.57 <sub><sup>(2.0e+02)</sup></sub> | 35.24 <sub><sup>(4.0e+02)</sup></sub> | 28.98 <sub><sup>(1.0e+03)</sup></sub> | 22.20 <sub><sup>(2.0e+03)</sup></sub> |
| SignOPT (2019) | 37.78 <sub><sup>(2.0e+02)</sup></sub> | 34.80 <sub><sup>(4.0e+02)</sup></sub> | 18.68 <sub><sup>(1.0e+03)</sup></sub> | 12.12 <sub><sup>(2.0e+03)</sup></sub> |
| HSJA (2019) | 40.82 <sub><sup>(2.2e+02)</sup></sub> | 29.00 <sub><sup>(4.3e+02)</sup></sub> | 14.88 <sub><sup>(9.8e+02)</sup></sub> | 9.36 <sub><sup>(1.8e+03)</sup></sub> |
| Stealthy OPT (2023) | 35.58 <sub><sup>(1.2e+04)</sup></sub> | 22.50 <sub><sup>(8.0e+05)</sup></sub> | 12.38 <sub><sup>(2.5e+06)</sup></sub> | 7.72 <sub><sup>(4.9e+06)</sup></sub> |
| Stealthy HSJA (2023) | **26.62** <sub><sup>(4.2e+05)</sup></sub> | **19.42** <sub><sup>(8.2e+05)</sup></sub> | **11.50** <sub><sup>(2.0e+06)</sup></sub> | **6.74** <sub><sup>(4.0e+06)</sup></sub> |

</p>

### $\ell_\infty$

<p align="center">

|                            | 100            | 200            | 500            | 1000           |
|:---------------------------|--------------:|--------------:|--------------:|--------------:|
| HSJA (2019) | 34.22 <sub><sup>(2.0e+02)</sup></sub> | 30.78 <sub><sup>(4.0e+02)</sup></sub> | 19.66 <sub><sup>(1.0e+03)</sup></sub> | 12.43 <sub><sup>(2.0e+03)</sup></sub> |
| RayS (2020) | 13.62 <sub><sup>(1.5e+02)</sup></sub> | 8.73 <sub><sup>(2.8e+02)</sup></sub> | 6.02 <sub><sup>(6.3e+02)</sup></sub> | **5.16** <sub><sup>(1.2e+03)</sup></sub> |
| Stealthy RayS (2023) | **8.16** <sub><sup>(1.1e+03)</sup></sub> | **6.63** <sub><sup>(1.3e+03)</sup></sub> | **5.99** <sub><sup>(1.7e+03)</sup></sub> | 5.87 <sub><sup>(2.4e+03)</sup></sub> |

</p>

## Abstract

> Decision-based evasion attacks repeatedly query a black-box classifier to generate adversarial examples. Prior work measures the cost of such attacks by the total number of queries made to the classifier. We argue this metric is flawed. Most security-critical machine learning systems aim to weed out "bad" data (e.g., malware, harmful content, etc). Queries to such systems carry a fundamentally *asymmetric cost*: queries detected as "bad" come at a higher cost because they trigger additional security filters, e.g., usage throttling or account suspension. Yet, we find that existing decision-based attacks issue a large number of "bad" queries, which likely renders them ineffective against security-critical systems. We then design new attacks that reduce the number of bad queries by $1.5$-$7.3\times$, but often at a significant increase in total (non-bad) queries. We thus pose it as an open problem to build black-box attacks that are more effective under realistic cost metrics.

## Environment setup

The code was developed using Python 3.10.6 and Conda. The environment can be reproduced by creating the Conda environment in [`environment.yml`](./environment.yml) via

```sh
conda env create --file=environment.yml
```

This should install all the libraries needed to run the code. Please note that there could be some issues due to the CUDA version, so make sure that your hardware software stack supports the CUDA versions specified in `environment.yml`. Otherwise, it's also possible to install the modules we use in a non-Conda environment by running

```sh
pip install -r requirements.txt
```

## Evaluate your own attack

To evaluate your own attack against our benchmarks, your attack should implement the `BaseAttack` interface, defined in [`src/attacks/base.py`](src/attacks/base.py). In particular, if it follows the HSJA scheme of optimizing a perturbation, we suggest to implement the `PerturbationAttack` interface, while if it follows the OPT scheme of optimizing a direction, we suggest to implement the `DirectionAttack` interface. Of course, your attack does not have to follow one of these schemes. When calling the model, you should make sure that you are using the `is_correct_boundary_side` method from `BaseAttack` to make sure that the queries counter is updated correctly. This method returns a tensor of booleans (`True` if the input is classified as *good* and `False` otherwise) and the updated queries counter (which is **not** updated in place to avoid side effects!). We show examples of we adapted how pre-existing attacks to this interface in the [`src/attacks`](src/attacks) directory.

Once you evaluated your own attack, you can submit it to our leaderboard by opening an issue or a pull request.

## Running the paper's experiments

The entry point to run the attacks is the main [`main.py`](main.py) file. In particular, the file supports several arguments to run different attacks with different parameters on different datasets.

## Datasets and models

- ImageNet: We attack `torchvision`'s ResNet50 `V1` checkpoint.
- Binary Imagenet: a binary version of ImageNet, where the positives are the "dogs" classes, and the negatives are the other classes; we attack a ResNet50 fine-tuned on this dataset for one epoch starting from `torchvision`'s `V2` checkpoint.
- ImageNet NSFW: a subset of ImageNet images classified as NSFW by LAION-AI's CLIP NSFW classifier; we attack this classifier, ported to PyTorch.

### Loading the models and the data

#### ImageNet

The model is automatically downloaded by `torchvision`, and the dataset should be in `torchvision` format, i.e., a directory. 

#### Binary ImageNet

The model can be downloaded from [here](https://github.com/ethz-privsec/realistic-adv-examples/releases/download/v0.1/binary_imagenet.ckpt), and should be placed in the [`checkpoints`](checkpoints) folder placed in the working directory from which the `main.py` script is launched. The training of the model can be reproduced with [this](/scripts/train_dogs_model.py) script. The dataset is generated on the fly from `torchvision`'s ImageNet.

#### ImageNet NSFW

The model can be downloaded from [here](https://github.com/ethz-privsec/realistic-adv-examples/releases/download/v0.1/clip_autokeras_nsfw_torch.pth), and should be placed in the [`checkpoints`](checkpoints) folder placed in the working directory from which the `main.py` script is launched. The porting of the classifier can be reproduced with [this](/scripts/port_keras_model.py) script.

> **Note**
> The porting script requires the installation of `tensorflow` and `autokeras`, which are not included in [`environment.yml`](environment.yml) nor in [`requirements.txt`](requirements.txt) to keep the environment lean. Moreover, it requires the download and uncompression of the Keras model inside of the [`checkpoints`](checkpoints) directory. Instructions for this can be found here on the [original repository](https://github.com/LAION-AI/CLIP-based-NSFW-Detector/).

The dataset is generated from `torchvision`'s ImageNet. The dataset is also automatically initialized and downloads the files needed to build the dataset. The files are downloaded to the `nsfw_imagenet` directory, which is created in the parent folder of where the `ImageNet` dataset is placed. The generation of the scores downloaded from GitHub can be reproduced with [this](scripts/compute_nsfw_outputs.py) script.

> **Note**
> The downloaded files **do not contain** any NSFW content. They are just ther outputs of the CLIP NSFW classifier as a NumPy array and filenames of the corresponding images. The content classified as NSFW is already contained in ImageNet itself.

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

## Credits and acknowledgments

The attack implementations have been adapted from the official implementations for [OPT](https://github.com/LeMinhThong/blackbox-attack), [SignOPT](https://github.com/cmhcbb/attackbox), [HSJA](https://github.com/Jianbo-Lab/HSJA/), and the Boundary Attack (released as part of [Foolbox](https://github.com/bethgelab/foolbox)). Some parts are also borrowed from the code of the paper [Preprocessors Matter! Realistic Decision-BasedAttacks on Machine Learning Systems](https://github.com/google-research/preprocessor-aware-black-box-attack). Finally, the CLIP NSFW classifier is ported by us to PyTorch from the Keras [version](https://github.com/LAION-AI/CLIP-based-NSFW-Detector/) released by LAION-AI, under MIT license.
