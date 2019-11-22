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
from metrics.inception_score import inception_score

sns.set(context="paper", style="white")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=util.HelpFormatter)

    parser.add_argument(
        "-data",
        type=str,
        default=None,
        required=True,
        help="Path to a .npy file with conditional probabilities.",
    )
    parser.add_argument(
        "-savefile",
        type=str,
        default=None,
        required=True,
        help="Name of the output file.",
    )
    parser.add_argument(
        "-no_timestamp",
        action="store_true",
        help="If set, disables adding '.time[timestamp]' to -savefile",
    )
    # Noise
    parser.add_argument(
        "-noise",
        type=str,
        choices=["blur", "gaussian", "sap", "swirl"],
        default=None,
        help="Type of noise to apply to test images. 'sap' is Salt & Pepper. NOTE: This only saves the provided value in the .npz archive!",
    )
    parser.add_argument(
        "-noise_amount",
        type=float,
        default=None,
        help="Standard deviation for blur, variance for noise, proportion of pixels "
        "for s&p, strength for swirl. NOTE: This only saves the provided value in the .npz archive!",
    )
    parser.add_argument(
        "-noise_radius",
        type=float,
        default=None,
        help="Radius of swirl. NOTE: This only saves the provided value in the .npz archive!",
    )
    # Labels
    parser.add_argument(
        "-labels",
        type=str,
        default=None,
        help="Path to a .npy file with labels for -data.",
    )
    parser.add_argument(
        "-allowed_labels",
        nargs="+",
        type=int,
        default=None,
        help="List of label IDs to pick from -labels. Data in -labels "
        "not matching these labels will be discarded.",
    )
    parser.add_argument(
        "-sample",
        type=int,
        default=None,
        help="If set, will randomly pick that many samples from the dataset.",
    )
    parser.add_argument("-seed", type=int, default=None, help="A seed for numpy.")

    args = parser.parse_args()

    np.random.seed(args.seed)

    os.makedirs(os.path.dirname(args.savefile), exist_ok=True)

    print("Loading Data")
    print("- data:", args.data)
    data = np.load(args.data)

    if args.sample:
        data = data[np.random.choice(data.shape[0], args.sample, replace=False)]

    original_data_dtype = data.dtype

    if np.ndim(data) > 2:
        data = np.reshape(data, (data.shape[0], -1))

    if not args.no_timestamp:
        args.savefile = f"{args.savefile}.time[{time.time()}]"
    print("Save Path:", args.savefile)

    if args.labels:
        if args.allowed_labels is None:
            parser.error("When -labels is set you have to also specify -allowed_labels")
        labels = np.load(args.labels).astype(np.int64)
        unique_labels = np.unique(labels)

        if labels.shape[0] != data.shape[0]:
            raise ValueError("Shape[0] of -labels and -data do not match")

        allowed_labels = np.unique(args.allowed_labels)
        if not np.isin(allowed_labels, unique_labels).all():
            raise ValueError("Not all -allowed_labels are in -labels")

        label_indices = np.where(np.isin(labels, allowed_labels))
        original_data_shape = data.shape
        data = data[label_indices]
        print(f"Selected {data.shape[0]} elements from -data")

    if args.noise:
        if args.noise_amount is None:
            parser.error("When -noise is set you have to also set -noise_amount")
        if args.noise == "swirl" and args.noise_radius is None:
            parser.error("When -noise=swirl you have to also set -noise_radius.")

    print("Data Statistics:")
    print("----------------")
    if args.labels:
        print("- Original Shape:", original_data_shape)
    print("- Shape:", data.shape)
    if original_data_dtype != data.dtype:
        print("- Original Dtype:", original_data_dtype)
    print("- Dtype:", data.dtype)
    print("- Min:", data.min())
    print("- Max:", data.max())
    print("- Labels:", "True" if args.labels else "False")

    score = inception_score(data)
    print("IS:", score)

    additional_save_data = {}
    if args.allowed_labels:
        additional_save_data["original_labels"] = unique_labels
        additional_save_data["allowed_labels"] = allowed_labels
    if args.noise:
        additional_save_data["noise"] = args.noise
        additional_save_data["noise_amount"] = args.noise_amount
        if args.noise == "swirl":
            additional_save_data["noise_radius"] = args.noise_radius
    if args.sample:
        additional_save_data["sample"] = args.sample

    np.savez_compressed(args.savefile, inception_score=score, **additional_save_data)

