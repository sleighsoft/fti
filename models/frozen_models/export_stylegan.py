# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config

import tensorflow as tf

def export_graphdef(filename):
    from tensorflow.python.tools import freeze_graph

    graph = tf.get_default_session().graph
    sess = tf.get_default_session()
    graph_def = graph.as_graph_def()

    # fix batch norm nodes
    for node in graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        
        inputs = []
        for inp in node.input:
            if inp[0] == '^':
                node.input.remove(inp)

    saver_path = tf.train.Saver().save(sess, 'cache/karras2019stylegan-ffhq-1024x1024.ckpt')
    converted_graph = tf.graph_util.convert_variables_to_constants(sess, graph_def, ['Gs/images_out'])
    tf.train.write_graph(converted_graph, 'cache', f'{filename}_converted.pb', as_text=False)
    graph_path = tf.train.write_graph(converted_graph, 'cache', f'{filename}.pbtxt')

    print('Freezing graph')
    freeze_graph.freeze_graph(
        input_graph=graph_path,
        input_saver='',
        input_binary=False,
        input_checkpoint=saver_path,
        output_node_names=['Gs/images_out'],
        restore_op_name='',
        filename_tensor_name='',
        output_graph=f'cache/frozen_{filename}.pb',
        clear_devices=False,
        initializer_nodes='',
        variable_names_whitelist="",
        variable_names_blacklist="",
        input_meta_graph=None,
        input_saved_model_dir=None
    )

def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
    with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
        _G, _D, Gs = pickle.load(f)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    Gs.print_layers()

    export_graphdef('karras2019stylegan-ffhq-1024x1024')

if __name__ == "__main__":
    main()
