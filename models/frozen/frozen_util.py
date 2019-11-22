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


import tarfile
import os
import urllib.request

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import map_fn as map_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer
from tensorflow.python.eager import def_function


def graph_def_from_url_targz(url, outfile, filename, silent=False):
    """Get a GraphDef proto from a URL to a .tar.gz archive.

    Saves the archive at `outfile` and will not re-download it if already
    present.

    Args:
        url: URL to .tar.gz archive.
        outfile: Name of the file where `url` is downloaded to.
        filename: Name of the GraphDef file inside the .tar.gz.

    Returns:
        A GraphDef proto.
    """
    if not os.path.exists(outfile):

        def _progress(count, block_size, total_size):
            dl_size = float(count * block_size) / float(total_size) * 100.0
            print(f"\r>> Downloading {outfile} {dl_size:.1f}", flush=True, end="")

        outfile, _ = urllib.request.urlretrieve(url, outfile, _progress)
        print(flush=True)

    with tarfile.open(outfile, "r:gz") as tar:
        proto_str = tar.extractfile(filename).read()
    return graph_pb2.GraphDef.FromString(proto_str)


def graph_def_from_disk_targz(targz_file, filename):
    """Get a GraphDef proto from a disk location.

    Args:
        targz_file: A .tar.gz containing a GraphDef.
        filename: Name of the GraphDef file inside the .tar.gz.

    Returns:
        A GraphDef proto.
    """
    with tarfile.open(targz_file, "r:gz") as tar:
        proto_str = tar.extractfile(filename).read()
    return graph_pb2.GraphDef.FromString(proto_str)


def run_from_graph_def(
    graph_def,
    input_name,
    output_name,
    input_tensor,
    dtype,
    batch_size=None,
    parallel_iterations=None,
):
    """Runs the GraphDef with `input_tensor` mapped to `input_name` and returns
    the tensor matching `output_name`.

    Args:
        graph_def: A GraphDef proto.
        input_name: Name of the input node in `graph_def`.
        output_name: Name of the output node in `graph_def`.
        input_tensor: Input tensor to the graph. A single input or a batch of
            inputs.
        dtype: Return type of `output_name` node.
        batch_size: (optional) The size per batch of input to the graph. If
            specified this will add a dynamically computed batch dimension to
            `input_tensor`. `input_tensor` must be a batch if this is set.
        parallel_iterations: (optional) The number of iterations allowed to
            run in parallel. When graph building, the default value is 10.
            While executing eagerly, the default value is set to 1.

    Returns: The output of the `graph_def` given `input_tensor` of type
        `dtype`. This will squeeze dimensions of output.

    Raises:
        InvalidArgumentError: If `batch_size` is set and `input_tensor` cannot
            be evenly split into batches.
    """
    output_names = [output_name]

    @def_function.function
    def fn(tensor):
        input_map = {input_name: tensor}
        classifier_outputs = importer.import_graph_def(
            graph_def, input_map, output_names
        )
        return classifier_outputs[0]

    @def_function.function
    def run(tensor):
        if batch_size is not None:
            with ops.device("/CPU:0"):
                shape = tensor.shape
                outshape = array_ops.concat(
                    [constant_op.constant([shape[0] // batch_size, -1]), shape[1:]], 0
                )
                tensor = gen_array_ops.reshape(tensor, outshape)

        outputs = map_ops.map_fn(
            fn=fn,
            elems=tensor,
            back_prop=False,
            swap_memory=True,
            dtype=dtype,
            parallel_iterations=parallel_iterations,
        )

        output = array_ops.concat(array_ops.unstack(outputs), 0)
        output = array_ops.squeeze(output)
        return output

    return run(input_tensor)
