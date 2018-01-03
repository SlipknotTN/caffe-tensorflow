import argparse
import os

import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph

from intermediate.model import Architecture


def doParsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, required=False, default=None, help='input size, don''t pass for None size')
    parser.add_argument('--inputChannels', type=int, required=False, default=3, help='Model input channels')
    parser.add_argument('--outputDir', type=str, required=True, help='Output directory where to save frozen graph')
    parser.add_argument('--dataPath', type=str, default='./intermediate/data.npy', help='Weights npy file')
    parser.add_argument('--inputNode', type=str, default="input", required=False, help="Name of the input node")
    parser.add_argument('--outputNodes', type=str, required=True, nargs='*',
                        help='The output nodes names. If your network has more than one output, stack them with a "," ')
    args = parser.parse_args()
    return args


def main():

    args = doParsing()
    print(args)

    # Input placeholder with any batch size, it is not defined in Architecture class.
    # It is possible to define any width and height (set None), but it does not work with current code as is
    # (e.g. SqueezeNet needs to convert latest 4D Tensor to 2D to get a working softmax)
    x = tf.placeholder(tf.float32, shape=[None, args.size, args.size, args.inputChannels], name=args.inputNode)

    # Create an instance of the net
    net = Architecture({args.inputNode: x})

    # Load weights from npy file
    with tf.Session() as sess:
        # Load the data
        net.load(args.dataPath, sess)
        # Save graph def and checkpoint
        saver = tf.train.Saver()
        checkpointPath = os.path.join(args.outputDir, 'model.chkpt')
        graphDefPath = os.path.join(args.outputDir, "model_graph_def.pb")
        saver.save(sess, checkpointPath)
        # This graph def will contains also weights, but we freeze the graph to complete the process in a clean way
        # (model ready for deployment, clear devices, etc...)
        tf.train.write_graph(sess.graph.as_graph_def(), "", graphDefPath, False)

    # Freeze the graph
    outputNodesNames = ",".join(args.outputNodes)
    freeze_graph(input_graph=graphDefPath, input_saver="", input_binary=True,
                 input_checkpoint=checkpointPath,
                 output_node_names=outputNodesNames,
                 restore_op_name="save/restore_all", filename_tensor_name="save/Const:0",
                 output_graph=os.path.join(args.outputDir, "graph.pb"), clear_devices=True, initializer_nodes="")


if __name__ == "__main__":
    main()
