from __future__ import absolute_import, division, print_function

import random
from pathlib import Path

import numpy as np
from datasets.load import load_dataset
from keras.utils import to_categorical
from transformers import CLIPProcessor


class ImageData():
    def __init__(self, dataset_name):
        if dataset_name == 'mnist':
            from keras.datasets import mnist
            (x_train, y_train), (x_val, y_val) = mnist.load_data()
            x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

            x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)

        elif dataset_name == 'cifar100':
            from keras.datasets import cifar100
            (x_train, y_train), (x_val, y_val) = cifar100.load_data()

        elif dataset_name == 'cifar10':
            from keras.datasets import cifar10

            # Load CIFAR10 Dataset
            (x_train, y_train), (x_val, y_val) = cifar10.load_data()

        x_train = x_train.astype('float32') / 255
        x_val = x_val.astype('float32') / 255

        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)

        x_train_mean = np.zeros(x_train.shape[1:])
        x_train -= x_train_mean
        x_val -= x_train_mean
        self.clip_min = 0.0
        self.clip_max = 1.0

        self.x_train = x_train
        self.x_val = x_val
        self.y_train = y_train
        self.y_val = y_val
        self.x_train_mean = x_train_mean


def load_imagenet_nsfw(indices_path: Path | None = None):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    im_mean = np.array(processor.feature_extractor.image_mean).reshape(1, 3, 1, 1)  # type: ignore
    im_std = np.array(processor.feature_extractor.image_std).reshape(1, 3, 1, 1)  # type: ignore

    def transform(batch):
        if "image" not in batch.keys():
            return batch
        preprocessed_images = processor(images=batch["image"], return_tensors="np", padding=True)["pixel_values"]
        unnormalized_images = np.round((preprocessed_images * im_std + im_mean) * 255) / 255
        batch["image"] = unnormalized_images
        return batch

    val_dataset = load_dataset("dedeswim/imagenet-nsfw", split="train")
    val_dataset = val_dataset.with_transform(transform)
    if indices_path is not None:
        print(f"Filtering datasets keeping indices in {indices_path}")
        indices = np.load(indices_path)
        val_dataset = val_dataset.select(indices)

    rand_seed = 42
    np.random.seed(rand_seed)
    random.seed(rand_seed)

    return val_dataset["image"], to_categorical(np.array(val_dataset["label"]))


def split_data(x, y, model, num_classes=10, split_rate=0.8, sample_per_class=100):
    # print('x.shape', x.shape)
    # print('y.shape', y.shape)

    np.random.seed(10086)
    pred = model.predict(x)
    label_pred = np.argmax(pred, axis=1)
    label_truth = np.argmax(y, axis=1)
    correct_idx = label_pred == label_truth
    print('Accuracy is {}'.format(np.mean(correct_idx)))
    x, y = x[correct_idx], y[correct_idx]
    label_pred = label_pred[correct_idx]

    x_train, x_test, y_train, y_test = [], [], [], []
    for class_id in range(num_classes):
        _x = x[label_pred == class_id][:sample_per_class]
        _y = y[label_pred == class_id][:sample_per_class]
        l = len(_x)
        x_train.append(_x[:int(l * split_rate)])
        x_test.append(_x[int(l * split_rate):])

        y_train.append(_y[:int(l * split_rate)])
        y_test.append(_y[int(l * split_rate):])

    x_train = np.concatenate(x_train, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    idx_train = np.random.permutation(len(x_train))
    idx_test = np.random.permutation(len(x_test))

    x_train = x_train[idx_train]
    y_train = y_train[idx_train]

    x_test = x_test[idx_test]
    y_test = y_test[idx_test]

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    import argparse

    from build_model import ImageModel
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, choices=['mnist', 'cifar10', 'cifar100'], default='mnist')

    parser.add_argument('--model_name', type=str, choices=['cnn', 'resnet', 'densenet'], default='cnn')

    args = parser.parse_args()
    dict_a = vars(args)

    data_model = args.dataset_name + args.model_name

    dataset = ImageData(args.dataset_name)

    model = ImageModel(args.model_name, args.dataset_name, train=False, load=True)

    x, y = dataset.x_val, dataset.y_val

    x_train, y_train, x_test, y_test = split_data(x, y, model, num_classes=10, split_rate=0.8)
