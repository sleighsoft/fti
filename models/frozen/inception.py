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
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import image_ops_impl as img_ops
from tensorflow.python.eager import def_function


from . import frozen_util


# Inception V3
# - Checkpoint: http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
# - Run in https://github.com/tensorflow/models/tree/master/research/slim
#   1. GraphDef: python export_inference_graph.py --model_name=inception_v3 --output_file=MODELS/inception_v3_inf_graph.pb
#   2. Obtain a copy of: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py
#   3. Frozen Graph: python freeze_graph.py --input_graph=MODELS/inception_v3_inf_graph.pb --input_checkpoint=CHECKPOINTS/inception_v3.ckpt --input_binary=true --output_graph=MODELS/frozen_inception_v3.pb --output_node_names=InceptionV3/Predictions/Reshape_1
INCEPTION_V3_URL = ""
INCEPTION_V3_OUTFILE = "frozen_models/frozen_inception_v3.tar.gz"
INCEPTION_V3_FILENAME = "frozen_inception_v3.pb"
# Same as Mul:0 of original model.
INCEPTION_V3_INPUT = "input:0"
INCEPTION_V3_INPUT_SHAPE = [None, 299, 299, 3]
# Same as pool_3:0 of original model.
INCEPTION_V3_OUTPUT = "InceptionV3/Logits/AvgPool_1a_8x8/AvgPool:0"
INCEPTION_V3_OUTPUT_SHAPE = [None, 1, 1, 2048]
# Softmax Output
INCEPTION_V3_OUTPUT_FINAL = "InceptionV3/Predictions/Reshape_1:0"
INCEPTION_V3_OUTPUT_SHAPE_FINAL = [None, 1001]


# Inception V4
# - Checkpoint: http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
# - Run in https://github.com/tensorflow/models/tree/master/research/slim
#   1. GraphDef: python export_inference_graph.py --model_name=inception_v4 --output_file=MODELS/inception_v4_inf_graph.pb
#   2. Obtain a copy of: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py
#   3. Frozen Graph: python freeze_graph.py --input_graph=MODELS/inception_v4_inf_graph.pb --input_checkpoint=MODELS/inception_v4.ckpt --input_binary=true --output_graph=MODELS/frozen_inception_v4.pb --output_node_names=InceptionV4/Logits/Logits/BiasAdd
INCEPTION_V4_URL = ""
INCEPTION_V4_OUTFILE = "frozen_inception_v4.tar.gz"
INCEPTION_V4_FILENAME = "frozen_inception_v4.pb"
INCEPTION_V4_INPUT = "input:0"
INCEPTION_V4_INPUT_SHAPE = [None, 299, 299, 3]
INCEPTION_V4_OUTPUT = "InceptionV4/Logits/AvgPool_1a/AvgPool:0"
INCEPTION_V4_OUTPUT_SHAPE = [None, 1, 1, 1536]


def versions():
    return ["v3", "v4", "v3-final"]


def version_info(version):
    if not version in versions():
        raise ValueError(f"Unsupported version {version}. Only {versions()} allowed")
    if version == "v3":
        return {
            "url": INCEPTION_V3_URL,
            "outfile": INCEPTION_V3_OUTFILE,
            "filename": INCEPTION_V3_FILENAME,
            "input": INCEPTION_V3_INPUT,
            "output": INCEPTION_V3_OUTPUT,
            "input_shape": INCEPTION_V3_INPUT_SHAPE,
            "output_shape": INCEPTION_V3_OUTPUT_SHAPE,
        }
    elif version == "v4":
        return {
            "url": INCEPTION_V4_URL,
            "outfile": INCEPTION_V4_OUTFILE,
            "filename": INCEPTION_V4_FILENAME,
            "input": INCEPTION_V4_INPUT,
            "output": INCEPTION_V4_OUTPUT,
            "input_shape": INCEPTION_V4_INPUT_SHAPE,
            "output_shape": INCEPTION_V4_OUTPUT_SHAPE,
        }
    elif version == "v3-final":
        return {
            "url": INCEPTION_V3_URL,
            "outfile": INCEPTION_V3_OUTFILE,
            "filename": INCEPTION_V3_FILENAME,
            "input": INCEPTION_V3_INPUT,
            "output": INCEPTION_V3_OUTPUT_FINAL,
            "input_shape": INCEPTION_V3_INPUT_SHAPE,
            "output_shape": INCEPTION_V3_OUTPUT_SHAPE_FINAL,
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

    Preprocessing as in: https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py#L253

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

    images = img_ops.convert_image_dtype(images, dtype=dtypes.float32)

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
    images -= 0.5
    images *= 2.0
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

    with ops.device("/CPU:0"):
        output = array_ops.squeeze(output)

    return output
