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


import numpy as np
from skimage.transform import swirl
from skimage.filters import gaussian
from skimage.util import random_noise


def apply_gaussian_blur(image, sigma):
    multichannel = None
    if np.ndim(image) == 3:
        if image.shape[2] == 1:
            multichannel = False
        else:
            multichannel = True
    return gaussian(image, sigma, multichannel=multichannel)


def apply_gaussian_noise(image, var):
    return random_noise(image, var=var)


def apply_salt_and_pepper_old(image, amount):
    return random_noise(image, mode="s&p", amount=amount)


def apply_salt_and_pepper(image, amount):
    new = np.empty_like(image, dtype=np.float64)
    seed = np.random.randint(np.iinfo(np.int32).max)
    if np.ndim(image) == 3:
        for channel in range(image.shape[-1]):
            new[:, :, channel] = random_noise(
                image[:, :, channel], mode="s&p", amount=amount, seed=seed
            )
    else:
        new[:, :] = random_noise(image, mode="s&p", amount=amount, seed=seed)
    return new


def apply_swirl(image, strength, radius):
    return swirl(image, strength=strength, radius=radius)
