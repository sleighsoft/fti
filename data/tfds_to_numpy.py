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
import time
import os

import numpy as np
import tensorflow_datasets as tfds

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def to_numpy(dataset, split, outdir, num_samples):
    image_file = os.path.join(outdir, f"{dataset}_{split}_N[{num_samples}].npy")
    label_file = os.path.join(outdir, f"{dataset}_{split}_labels_N[{num_samples}].npy")
    if not (os.path.exists(image_file) and os.path.exists(label_file)):
        print(f"Saving {dataset} {split} images as numpy")

        tf_dataset, info = tfds.load(
            name=dataset, split=split, with_info=True, shuffle_files=False
        )

        assert num_samples <= info.splits[split].num_examples

        if num_samples == -1:
            num_samples = info.splits[split].num_examples

        tf_dataset = tf_dataset.take(num_samples)

        images = np.empty(
            (num_samples, *info.features["image"].shape),
            dtype=info.features["image"].dtype.as_numpy_dtype(),
        )
        labels = np.empty(
            num_samples, dtype=info.features["label"].dtype.as_numpy_dtype()
        )

        for i, pair in enumerate(tf_dataset):
            image = pair["image"]
            label = pair["label"]
            images[i] = image
            labels[i] = label

        # Save images
        np.save(image_file, images)
        # Save labels
        np.save(label_file, labels)
    else:
        print(f"{image_file} and {label_file} already exists. Skipping.")


def main(outdir, dataset, num_samples, splits):
    os.makedirs(outdir, exist_ok=True)

    for split in splits:
        to_numpy(dataset, split, outdir, num_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("outdir", type=str, help="Directory to write data to.")
    parser.add_argument(
        "dataset", type=str, help="Dataset to use. One of tensorflow datasets."
    )
    parser.add_argument(
        "-num_samples",
        type=int,
        default=-1,
        help="Number of samples to transform. -1 = all",
    )
    parser.add_argument("-splits", nargs="+", type=str)

    args = parser.parse_args()

    main(args.outdir, args.dataset, args.num_samples, args.splits)
