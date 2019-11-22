# The following models are currently supported:
- inception.py
  - frozen_inception_v3.tar.gz
  - frozen_inception_v4.tar.gz
- vgg.py
  - frozen_vgg16.tar.gz

# Model archives can be obtained the following way:

## Tensorflow SLIM models

1. Go to https://github.com/tensorflow/models/tree/master/research/slim
2. Clone the repository
3. Navigate to `research/slim/`.
4. Setup tensorflow 1.x (1.13.1 was used) (Note, the obtained graph can be used by 2.x)
5. Run the following to get the graph_def protobufs
   ```
   # For inception_v3
   python export_inference_graph.py --model_name=inception_v3 --output_file=inception_v3_inf_graph.pb
   # For inception_v4
   python export_inference_graph.py --model_name=inception_v4 --output_file=inception_v4_inf_graph.pb
   # For vgg16
   python export_inference_graph.py --model_name=vgg_16 --output_file=MODELS/vgg16_inf_graph.pb --labels_offset=1
   ```
6. Use `freeze_graph.py` to combine graph_def with checkpointed variables (a copy of freeze_graph.py is included in the `models/frozen_moodels` directory). Copy it into `research/slim/`.
7. Download checkpointed variables. Note, place donwloaded checkpoint files in the same directory as the `.pb` file (`reasearch/slim/`).
   * inception_v3: http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
   * inception_v4: http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
   * vgg_16: http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
8. Combine inference graph with checkpointed variables
   ```
   # For inception_v3
   python freeze_graph.py --input_graph=inception_v3_inf_graph.pb --input_checkpoint=inception_v3.ckpt --input_binary=true --output_graph=frozen_inception_v3.pb --output_node_names=InceptionV3/Predictions/Reshape_1
   # For inception_v4
   python freeze_graph.py --input_graph=MODELS/inception_v4_inf_graph.pb --input_checkpoint=MODELS/inception_v4.ckpt --input_binary=true --output_graph=MODELS/frozen_inception_v4.pb --output_node_names=InceptionV4/Logits/Logits/BiasAdd
   # For vgg16
   python .\freeze_graph.py --input_graph .\MODELS\vgg16_inf_graph.pb --input_checkpoint .\MODELS\vgg_16.ckpt --input_binary=True --output_graph .\MODELS\frozen_vgg16.pb --output_node_names=vgg_16/fc8/squeezed
   ```
9. Pack the `frozen_*.pb` into a `.tar.gz` archive matching the filenames mentioned at the top of this page.
10. Copy the obtained `.tar.gz` archive into this repository's `models/frozen_models/` directory.


Alternatively, prebuilt frozen model archives used by this work can be found here:
- https://1drv.ms/u/s!AsT4o_K0zoLQh9xOso6zJ2VsF2AlzQ?e=j5bdPH