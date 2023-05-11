# Evading Black-box Classifiers Without Breaking Eggs

Code to reproduce results of the submission *"Evading Black-box Classifiers Without Breaking Eggs"*.

## Leaderboard

Leaderboards which show, for each attack, the distances achieved after 100, 200, 500, and 1000 bad queries, and how many total queries have been issued at that amount of bad queries. We sort the attacks by publication date and bold the best result for each column. The leaderboard refers to untargeted attacks against an ImageNet ResNet-50. The results are presented in the form `distance`/`total_queries`. The $\ell_2$ norm is for images in the $[0, 1]$ range, and the $\ell_\infty$ one is for images in the $[0, 255]$ one.

### $\ell_2$

|                         | 100            | 200            | 500            | 1000           |
|:------------------------|:--------------:|:--------------:|:--------------:|:--------------:|
| OPT (Jul 2018)          | 37.57/1.99e+02 | 35.24/3.97e+02 | 28.98/1.01e+03 | 22.20/2.01e+03 |
| HSJA (Jun 2019)         | 40.82/2.17e+02 | 29.00/4.33e+02 | 14.88/9.85e+02 | 9.36/1.81e+03  |
| SignOPT (Sep 2019)      | 37.78/2.00e+02 | 34.80/3.96e+02 | 18.68/1.02e+03 | 12.12/1.96e+03 |
| Stealthy OPT (May 2023) | **35.58**/1.24e+04 | **22.50**/8.02e+05 | **12.38**/2.51e+06 | **7.72**/4.89e+06  |

### $\ell_\infty$

|                            | 100            | 200            | 500            | 1000           |
|:---------------------------|:--------------:|:--------------:|:--------------:|:--------------:|
| HSJA (Apr 2019)            | 34.22/2.01e+02 | 30.78/4.02e+02 | 19.66/9.99e+02 | 12.43/2.00e+03 |
| RayS (Jun 2020)            | 13.62/1.50e+02 | 8.73/2.76e+02  | 6.02/6.33e+02  | **5.16**/1.21e+03  |
| Stealthy RayS (May 2023)   | **8.16**/1.13e+03  | **6.63**/1.26e+03  | **5.99**/1.68e+03  | 5.87/2.41e+03  |

## Environment

The code was developed using Python 3.10.6 and Conda. The environment can be reproduced by creating the Conda environment in [`environment.yml`](./environment.yml) via

```sh
conda env create --file=environment.yml
```

This should install all the libraries needed to run the code. Please note that there could be some issues due to the CUDA version, so make sure that your hardware software stack supports the CUDA versions specified in `environment.yml`. Otherwise, it's also possible to install the modules we use in a non-Conda environment by running

```sh
pip install -r requirements.txt
```

## Running the experiments

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
