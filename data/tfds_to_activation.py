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


import warnings

warnings.filterwarnings("ignore")
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.python.platform import tf_logging as logging

logging.set_verbosity(logging.ERROR)

import argparse
import time
from functools import partial
from importlib import import_module

import numpy as np
import tensorflow_datasets as tfds
from skimage.util import img_as_ubyte

from misc import images


def import_frozen(frozen_network):
    return import_module(f"models.frozen.{frozen_network}")


def to_activation(
    dataset,
    split,
    outdir,
    num_samples,
    image_batch_size,
    network_batch_size,
    frozen_network,
    frozen_network_version,
    no_labels=False,
    noise_fn=None,
):
    activation_file = os.path.join(
        outdir,
        f"{dataset}_{split}_to_{frozen_network}-{frozen_network_version}_N[{num_samples}].npy",
    )
    label_file = os.path.join(outdir, f"{dataset}_{split}_labels_N[{num_samples}].npy")

    if not os.path.exists(activation_file):
        print_prefix = f"Transforming {dataset} {split} images to activations"
        print(print_prefix, end="", flush=True)
        s_time = time.time()

        tf_dataset, info = tfds.load(
            name=dataset, split=split, with_info=True, shuffle_files=False
        )

        assert num_samples <= info.splits[split].num_examples

        if num_samples == -1:
            num_samples = info.splits[split].num_examples

        tf_dataset = tf_dataset.take(num_samples)
        tf_dataset = tf_dataset.batch(image_batch_size)

        frozen = import_frozen(frozen_network)

        embedded_images = None

        has_labels = "label" in info.features
        if has_labels and not no_labels:
            labels = np.empty(
                [num_samples], dtype=info.features["label"].dtype.as_numpy_dtype()
            )

        start = 0
        print(f"\r{print_prefix} ... {start}/{num_samples}", end="", flush=True)
        for pair in tf_dataset:
            s_batch_time = time.time()
            image_batch = pair["image"]

            if noise_fn:
                with_noise = noise_fn(image_batch.numpy())
                if image_batch.dtype == np.uint8:
                    if (with_noise < 0.0).any() or (with_noise > 1.0).any():
                        print(
                            "Warning: Detected original images are uint8 but noised images are outside of range [0, 1]. Clipping values!"
                        )
                        np.clip(with_noise, 0.0, 1.0, out=with_noise)
                    image_batch = img_as_ubyte(with_noise)
                else:
                    print(
                        "Warning: Images are not uint8. Applying noise will convert images to float64!"
                    )
                    image_batch = with_noise

            batch_size = image_batch.shape[0]

            # Handle last batch sometimes being smaller than network_batch_size
            current_network_batch_size = network_batch_size
            while batch_size % current_network_batch_size != 0:
                current_network_batch_size -= 1

            activations = frozen.run(
                frozen_network_version, image_batch, current_network_batch_size
            )

            batch_size = activations.shape[0]

            if start == 0:
                embedded_images = np.empty(
                    [num_samples, activations.shape[-1]],
                    dtype=activations.dtype.as_numpy_dtype(),
                )

            embedded_images[start : start + batch_size, ...] = activations

            if has_labels and not no_labels:
                label_batch = pair["label"]
                labels[start : start + batch_size] = label_batch

            start += batch_size
            print(
                f"\r{print_prefix} ... {start}/{num_samples} "
                f"{batch_size / (time.time()-s_batch_time):.2f} samples/s",
                end="",
                flush=True,
            )

        print(
            f"\rTransformed {num_samples} {dataset} {split} images in "
            f"{time.time()-s_time:.2f}s"
        )
        # Save activations
        np.save(activation_file, embedded_images)
        if has_labels and not no_labels:
            # Save labels
            np.save(label_file, labels)
    else:
        print(f"{activation_file} and {label_file} already exists. Skipping.")


def main(
    outdir,
    dataset,
    num_samples,
    network_batch_size,
    image_batch_size,
    splits,
    frozen_network,
    frozen_network_version,
    no_labels=False,
    noise_fn=None,
):
    os.makedirs(outdir, exist_ok=True)

    for split in splits:
        to_activation(
            dataset,
            split,
            outdir,
            num_samples,
            image_batch_size,
            network_batch_size,
            frozen_network,
            frozen_network_version,
            no_labels,
            noise_fn,
        )


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
    parser.add_argument(
        "-network_batch_size",
        type=int,
        default=50,
        help="Batch size to use for networks.",
    )
    parser.add_argument(
        "-image_batch_size",
        type=int,
        default=5000,
        help="Batch size used for creating or loading images.",
    )
    parser.add_argument(
        "-no_labels", action="store_true", help="If set, do not export labels."
    )

    # Noise
    parser.add_argument(
        "-noise",
        type=str,
        choices=["blur", "gaussian", "sap", "swirl"],
        default=None,
        help="Type of noise to apply to test images. 'sap' is Salt & Pepper.",
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

    parser.add_argument("-splits", nargs="+", type=str)

    parser.add_argument("-frozen_network", type=str, default="inception")
    parser.add_argument("-frozen_network_version", type=str, default="v3")

    args = parser.parse_args()

    if args.noise is not None and args.noise_amount is None:
        parser.error("When -noise is set you have to also set -noise_amount.")

    if args.noise == "blur":
        noise_fn = partial(images.apply_gaussian_blur, sigma=args.noise_amount)
    elif args.noise == "gaussian":
        noise_fn = partial(images.apply_gaussian_noise, var=args.noise_amount)
    elif args.noise == "sap":
        noise_fn = partial(images.apply_salt_and_pepper, amount=args.noise_amount)
    elif args.noise == "swirl":
        if args.noise_radius is None:
            parser.error("When -noise=swirl you have to also set -noise_radius.")
        noise_fn = partial(
            images.apply_swirl, strength=args.noise_amount, radius=args.noise_radius
        )
    else:
        noise_fn = None

    main(
        args.outdir,
        args.dataset,
        args.num_samples,
        args.network_batch_size,
        args.image_batch_size,
        args.splits,
        args.frozen_network,
        args.frozen_network_version,
        args.no_labels,
        noise_fn,
    )
