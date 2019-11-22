# Copyright 2019 Julian Niedermeier & Goncalo Mordido
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
import time

import matplotlib.pyplot as plt
import seaborn as sns
from skimage.util.dtype import img_as_float64
import numpy as np

from misc import images, util
from metrics.frechet_inception_distance import frechet_inception_distance

sns.set(context="paper", style="white")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=util.HelpFormatter)

    parser.add_argument(
        "-train",
        type=str,
        default=None,
        required=True,
        help="Path to a .npy file with train dataset.",
    )
    parser.add_argument(
        "-test",
        type=str,
        default=None,
        required=True,
        help="Path to a .npy file with test dataset.",
    )
    parser.add_argument(
        "-savefile",
        type=str,
        default=None,
        required=True,
        help="Name of the output file.",
    )
    # Misc
    parser.add_argument(
        "-save_images",
        action="store_true",
        help="If set, will save test and train image originals and noised versions.",
    )
    parser.add_argument(
        "-no_timestamp",
        action="store_true",
        help="If set, disables adding '.time[timestamp]' to -savefile",
    )
    parser.add_argument(
        "-dont_use_noise_args",
        action="store_true",
        help="If set, will not use all the noise arguments to add noise to the input data but will save the argument values in the .npz archive.",
    )
    # Noise
    parser.add_argument(
        "-noise_target",
        type=str,
        choices=["train", "test"],
        default=None,
        help="To which data the noise is applied.",
    )
    parser.add_argument(
        "-noise",
        type=str,
        choices=["blur", "gaussian", "sap", "swirl"],
        default=None,
        help="Type of noise to apply to test images. 'sap' is Salt & Pepper.",
    )
    parser.add_argument(
        "-image_shape",
        type=str,
        default=None,
        help="Required if noise is set. Specifies width,height,channel.",
    )
    parser.add_argument(
        "-noise_amount",
        type=float,
        default=None,
        help="Standard deviation for blur, variance for noise, proportion of pixels "
        "for s&p, strength for swirl.",
    )
    parser.add_argument(
        "-noise_radius", type=float, default=None, help="Radius of swirl."
    )
    # Labels
    parser.add_argument(
        "-train_labels",
        type=str,
        default=None,
        help="Path to a .npy file with labels for -train.",
    )
    parser.add_argument(
        "-allowed_train_labels",
        nargs="+",
        type=int,
        default=None,
        help="List of label IDs to pick from -train_labels. Data in -train_labels "
        "not matching these labels will be discarded.",
    )
    parser.add_argument(
        "-test_labels",
        type=str,
        default=None,
        help="Path to a .npy file with labels for -test.",
    )
    parser.add_argument(
        "-allowed_test_labels",
        nargs="+",
        type=int,
        default=None,
        help="List of label IDs to pick from -test_labels. Data in -test_labels "
        "not matching these labels will be discarded.",
    )
    parser.add_argument(
        "-sample_train",
        type=int,
        default=None,
        help="If set, will randomly pick that many samples from the train set.",
    )
    parser.add_argument(
        "-sample_test",
        type=int,
        default=None,
        help="If set, will randomly pick that many samples from the test set.",
    )
    parser.add_argument("-seed", type=int, default=None, help="A seed for numpy.")

    args = parser.parse_args()

    np.random.seed(args.seed)

    os.makedirs(os.path.dirname(args.savefile), exist_ok=True)

    print("Loading Data")
    print("- train:", args.train)
    print("- test:", args.test)
    train = np.load(args.train)
    test = np.load(args.test)

    if args.sample_train:
        train = train[
            np.random.choice(train.shape[0], args.sample_train, replace=False)
        ]
    if args.sample_test:
        test = test[np.random.choice(test.shape[0], args.sample_test, replace=False)]

    original_train_dtype = train.dtype
    original_test_dtype = test.dtype
    assert train.dtype == test.dtype
    assert train.shape[1:] == test.shape[1:]

    if np.ndim(train) > 2:
        train = np.reshape(train, (train.shape[0], -1))
    if np.ndim(test) > 2:
        test = np.reshape(test, (test.shape[0], -1))

    if not args.no_timestamp:
        args.savefile = f"{args.savefile}.time[{time.time()}]"
    print("Save Path:", args.savefile)

    if args.train_labels:
        if args.allowed_train_labels is None:
            parser.error(
                "When -train_labels is set you have to also specify -allowed_train_labels"
            )
        train_labels = np.load(args.train_labels).astype(np.int64)
        unique_train_labels = np.unique(train_labels)

        if train_labels.shape[0] != train.shape[0]:
            raise ValueError(
                f"Shape[0] of -train_labels {train_labels.shape[0]} and -train {train.shape[0]} do not match"
            )

        allowed_train_labels = np.unique(args.allowed_train_labels)
        if not np.isin(allowed_train_labels, unique_train_labels).all():
            raise ValueError("Not all -allowed_train_labels are in -train_labels")

        train_label_indices = np.where(np.isin(train_labels, allowed_train_labels))
        original_train_shape = train.shape
        train = train[train_label_indices]
        print(f"Selected {train.shape[0]} elements from -train")

    if args.test_labels:
        if args.allowed_test_labels is None:
            parser.error(
                "When -test_labels is set you have to also specify -allowed_test_labels"
            )
        test_labels = np.load(args.test_labels).astype(np.int64)
        unique_test_labels = np.unique(test_labels)

        if test_labels.shape[0] != test.shape[0]:
            raise ValueError("Shape[0] of -test_labels and -test do not match")

        allowed_test_labels = np.unique(args.allowed_test_labels)
        if not np.isin(allowed_test_labels, unique_test_labels).all():
            raise ValueError("Not all -allowed_test_labels are in -test_labels")

        test_label_indices = np.where(np.isin(test_labels, allowed_test_labels))
        original_test_shape = test.shape
        test = test[test_label_indices]
        print(f"Selected {test.shape[0]} elements from -test")

    if args.noise:
        if args.noise_amount is None:
            parser.error("When -noise is set you have to also set -noise_amount")
        if args.noise == "swirl" and args.noise_radius is None:
            parser.error("When -noise=swirl you have to also set -noise_radius.")

    if args.noise and not args.dont_use_noise_args:
        if args.noise_target == "train":
            target = train
            other_target = "test"
            other = test
        else:
            target = test
            other_target = "train"
            other = train

        if not (other.min() >= -1.0 and other.max() <= 1.0):
            if other.dtype == np.uint8:
                other = img_as_float64(other)
            else:
                other /= 255.0
            if not (other.min() >= -1.0 and other.max() <= 1.0):
                raise ValueError(
                    f"{other_target} data cannot be normalized to range [-1, 1]"
                )

        if args.image_shape is None:
            parser.error("When -noise is set you have to also set -image_shape.")

        w, h, c = [int(n) for n in args.image_shape.split(",")]

        print(f"Distorting {args.noise_target} Images")
        distorted_images = np.empty(target.shape, dtype=np.float64)
        image_shape = (w, h) if c == 1 else (w, h, c)
        cmap = None if c > 1 else "gray"
        for i, image in enumerate(target):
            image = image.reshape(image_shape)
            if i == 0:
                if args.save_images:
                    plt.imshow(image, cmap=cmap)
                    plt.savefig(f"{args.savefile}.{args.noise_target}_original.png")

                    plt.imshow(other[0].reshape(image_shape), cmap=cmap)
                    plt.savefig(f"{args.savefile}.{other_target}_original.png")

            if args.noise == "blur":
                image = images.apply_gaussian_blur(image, args.noise_amount)
            elif args.noise == "gaussian":
                image = images.apply_gaussian_noise(image, args.noise_amount)
            elif args.noise == "sap":
                image = images.apply_salt_and_pepper(image, args.noise_amount)
            else:
                image = images.apply_swirl(image, args.noise_amount, args.noise_radius)

            if i == 0 and args.save_images:
                plt.imshow(image, cmap=cmap)
                plt.savefig(f"{args.savefile}.{args.noise_target}.png")

            distorted_images[i] = image.reshape(-1)

        if args.noise_target == "train":
            train = distorted_images
            test = other
        else:
            test = distorted_images
            train = other
    elif (
        (train.min() >= 0 and train.max() <= 255)
        and (test.min() >= 0 and test.max() <= 255)
        and train.dtype == np.uint8
    ):
        print("Data could be uint8 images. Converting to float64 in range [0,1]")
        train = img_as_float64(train)
        test = img_as_float64(test)
    elif train.dtype == np.float32 or test.dtype == np.float32:
        print("Detected train or test float32. Casting both to float64")
        train = train.astype(np.float64)
        test = test.astype(np.float64)

    print("Data Statistics:")
    print("----------------")
    print("Train")
    if args.train_labels:
        print("- Original Shape:", original_train_shape)
    print("- Shape:", train.shape)
    if original_train_dtype != train.dtype:
        print("- Original Dtype:", original_train_dtype)
    if args.allowed_train_labels is not None:
        print("- Allowed Labels:", allowed_train_labels)
    print("- Dtype:", train.dtype)
    print("- Min:", train.min())
    print("- Max:", train.max())
    print("- Noise:", "True" if args.noise_target == "train" else "False")
    print("- Labels:", "True" if args.train_labels else "False")
    print("Test")
    if args.test_labels:
        print("- Original Shape:", original_test_shape)
    print("- Shape:", test.shape)
    if original_test_dtype != test.dtype:
        print("- Original Dtype:", original_test_dtype)
    if args.allowed_test_labels is not None:
        print("- Allowed Labels:", allowed_test_labels)
    print("- Dtype:", test.dtype)
    print("- Min:", test.min())
    print("- Max:", test.max())
    print("- Noise:", "True" if args.noise_target == "test" else "False")
    print("- Labels:", "True" if args.test_labels else "False")

    try:
        fid = frechet_inception_distance(train, test)
    except Exception as e:
        print(e, "-> Setting FID to nan")
        fid = np.nan

    print("FID:", fid)

    additional_save_data = {}
    if args.allowed_train_labels:
        additional_save_data["original_train_labels"] = unique_train_labels
        additional_save_data["allowed_train_labels"] = allowed_train_labels
    if args.allowed_test_labels:
        additional_save_data["original_test_labels"] = unique_test_labels
        additional_save_data["allowed_test_labels"] = allowed_test_labels
    if args.noise_target:
        additional_save_data[f"{args.noise_target}_noise"] = args.noise
        additional_save_data[f"{args.noise_target}_noise_amount"] = args.noise_amount
        if args.noise == "swirl":
            additional_save_data[
                f"{args.noise_target}_noise_radius"
            ] = args.noise_radius
    if args.sample_train:
        additional_save_data["sample_train"] = args.sample_train
    if args.sample_test:
        additional_save_data["sample_test"] = args.sample_test

    np.savez_compressed(args.savefile, fid=fid, **additional_save_data)
