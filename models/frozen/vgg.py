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


from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import image_ops_impl as img_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op


from . import frozen_util


# VGG 16
# - Checkpoint: http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
# - Run in https://github.com/tensorflow/models/tree/master/research/slim
#   1. GraphDef: python export_inference_graph.py --model_name=vgg_16 --output_file=MODELS/vgg16_inf_graph.pb --labels_offset=1
#   2. Obtain a copy of: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py
#   3. Frozen Graph: python .\freeze_graph.py --input_graph .\MODELS\vgg16_inf_graph.pb --input_checkpoint .\MODELS\vgg_16.ckpt --input_binary=True --output_graph .\MODELS\frozen_vgg16.pb --output_node_names=vgg_16/fc8/squeezed
VGG16_URL = ""
VGG16_OUTFILE = "frozen_models/frozen_vgg16.tar.gz"
VGG16_FILENAME = "frozen_vgg16.pb"
VGG16_INPUT = "input:0"
VGG16_INPUT_SHAPE = [None, 224, 224, 3]
# Same node as in https://arxiv.org/abs/1904.06991
VGG16_OUTPUT = "vgg_16/fc7/Relu:0"
VGG16_OUTPUT_SHAPE = [None, 1, 1, 4096]

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def versions():
    return ["16"]


def version_info(version):
    if not version in versions():
        raise ValueError(f"Unsupported version {version}. Only {versions()} allowed")
    if version == "16":
        return {
            "url": VGG16_URL,
            "outfile": VGG16_OUTFILE,
            "filename": VGG16_FILENAME,
            "input": VGG16_INPUT,
            "output": VGG16_OUTPUT,
            "input_shape": VGG16_INPUT_SHAPE,
            "output_shape": VGG16_OUTPUT_SHAPE,
        }


def graph_def(version):
    info = version_info(version)
    gd = frozen_util.graph_def_from_url_targz(
        info["url"], info["outfile"], info["filename"]
    )
    return gd


def image_to_input_size(version, images, antialias=False):
    """Resizes images to size `version_input_shape`. This will convert
    grayscale to rgb.

    Note: Should be run on CPU.

    Preprocessing as in: https://github.com/tensorflow/models/blob/1af55e018eebce03fb61bba9959a04672536107d/research/slim/preprocessing/vgg_preprocessing.py#L319

    Note: Does not preserve aspect ratio if resize is necessary.

    Args:
        version: A supported inception version. See `versions()`.
        images: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D
            Tensor of shape `[height, width, channels]`. Where channels can
            be `1` or `3`.
        antialias: Whether to use an anti-aliasing filter when downsampling
            an image.

    Returns:
        A tensor of shape `[batch] + version_input_shape` or
        `version_input_shape`.
    """
    channels = images.shape[-1]
    assert channels == 1 or channels == 3

    info = version_info(version)
    batch_input_shape = info["input_shape"]
    input_shape = info["input_shape"][1:]

    if not (
        images.shape.is_compatible_with(batch_input_shape)
        or images.shape.is_compatible_with(input_shape)
    ):
        images = img_ops.resize_images_v2(
            images,
            size=input_shape[0:2],
            preserve_aspect_ratio=False,
            antialias=antialias,
        )
        if channels == 1:
            rank = array_ops.rank(images) - 1
            tile_shape = array_ops.concat(
                [array_ops.ones([rank], dtype=dtypes.int32), [3]], 0
            )
            images = gen_array_ops.tile(images, tile_shape)
    images = math_ops.cast(images, dtype=dtypes.float32)
    images -= array_ops.reshape(
        constant_op.constant([_R_MEAN, _G_MEAN, _B_MEAN]), [1, 1, 3]
    )
    return images


GRAPH_DEF = None


@def_function.function
def run(
    version, input_tensor, batch_size=None, parallel_iterations=None, antialias=False
):
    info = version_info(version)

    global GRAPH_DEF
    if GRAPH_DEF is None:
        GRAPH_DEF = graph_def(version)

    with ops.device("/CPU:0"):
        input_tensor = image_to_input_size(version, input_tensor, antialias)

    output = frozen_util.run_from_graph_def(
        GRAPH_DEF,
        info["input"],
        info["output"],
        input_tensor,
        dtypes.float32,
        batch_size=batch_size,
        parallel_iterations=parallel_iterations,
    )

    output = array_ops.squeeze(output)

    return output
