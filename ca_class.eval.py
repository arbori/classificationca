#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for CA Classification, built with tf.layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

print("Convolutional Neural Network Estimator for CA Classification, built with tf.layers")
print("Importing...")

import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

import random as rnd

import cellularautomata as ca
import classificationca as cca
import neigbor_spectrum as nspec

#tf.logging.set_verbosity(tf.logging.INFO)

###############################################################################
def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # Temporal Evolution are 100x100 cells, and have one color channel (black or white)
    input_layer = tf.reshape(features["x"], [-1, 100, 100, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 3x3 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 100, 100, 1]
    # Output Tensor Shape: [batch_size, 100, 100, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32, 
        kernel_size=[3, 3], 
        padding="same", 
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 100, 100, 32]
    # Output Tensor Shape: [batch_size, 50, 50, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 3x3 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 50, 50, 32]
    # Output Tensor Shape: [batch_size, 50, 50, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64, 
        kernel_size=[3, 3], 
        padding="same", 
        activation=tf.nn.relu)

    # Pooling Layer #2
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 50, 50, 64]
    # Output Tensor Shape: [batch_size, 25, 25, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Convolutional Layer #3
    # Computes 128 features using a 3x3 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 25, 25, 64]
    # Output Tensor Shape: [batch_size, 25, 25, 128]
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[3, 3], 
        padding="same", 
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 25, 25, 128]
    # Output Tensor Shape: [batch_size, 12, 12, 128]
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 12, 12, 128]
    # Output Tensor Shape: [batch_size, 12 * 12 * 128]
    pool3_flat = tf.reshape(pool3, [-1, 12 * 12 * 128])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 12 * 12 * 128]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    ### Why drouput elements?
        #==> Dropout: A Simple Way to Prevent Neural Networks from Overfitting (pdf)
        ### So, how define the rate?
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 4]
    logits = tf.layers.dense(inputs=dropout, units=4)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

###############################################################################
def classify_space_rule(k, r, total_ics, model_path, output_file, output_encoding='utf-8'):
    size = 100
    t = 3*size
    transient = 2*size

    m = 2*r + 1
    rule_size = int(pow(k, pow(k, m)))

    # Create the Estimator
    print("Create the Estimator")
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=model_path)

    # Predict CA in ray 1 space
    with open(output_file, mode='w', encoding=output_encoding) as a_file:
        for n in range(rule_size):
            print("Rule:", n)

            eval_data = cca.make_temporal_evolutions_rule(total_ics, n, k, r, t, transient)

            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": np.array(eval_data[0], dtype=np.float32)},
                num_epochs=1,
                shuffle=False)

            result_predict = mnist_classifier.predict(eval_input_fn, predict_keys=None, hooks=None)

            eval_prob = [p['probabilities'] for p in result_predict]
            eval_prob = [np.sum(line) for line in np.transpose(eval_prob)]
            eval_prob = eval_prob / np.sum(eval_prob)

            a_file.write(str(n))
            a_file.write("\t")
            a_file.write(str(cca.wolfram_class(n)))
            a_file.write("\t")
            a_file.write(str(np.argmax(eval_prob) + 1))
            a_file.write("\t")
            for val in eval_prob:
                a_file.write(str(val))
                a_file.write("\t")
            a_file.write("\n")

            a_file.flush()

###############################################################################
def network_predict_classification(temporal_evolution, classifier):
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(temporal_evolution, dtype=np.float32)},
        num_epochs=1,
        shuffle=False)

    result_predict = classifier.predict(eval_input_fn, predict_keys=None, hooks=None)

    eval_prob = [p['probabilities'] for p in result_predict]
    eval_prob = [np.sum(line) for line in np.transpose(eval_prob)]
    eval_prob = eval_prob / np.sum(eval_prob)

    return (np.argmax(eval_prob) + 1)

###############################################################################
def predict_with_network_and_spectrum(k, r, total_ics, size, model_path, database_file, output_file, output_encoding='utf-8'):
    '''
    '''
    t = 3*size
    transient = 2*size

    m = 2*r + 1
    rule_size = int(pow(k, pow(k, m)))

    # Create the Estimator
    print("Retriving the estimator in", model_path)
    network_estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=model_path)

    print("Retriving database", database_file)
    db = nspec.retrieve_spectrum_dataset(database_file)

    print("Creating initial conditions dataset...")
    ics = cca.init_conditions_numbers(k, t, total_ics)

    with open(output_file, mode='w', encoding=output_encoding) as outf:
        print("Create output file", output_file)
        outf.write("rule\tagree\ts1\ts2\ts3\ts4\tnet1\tnet2\tnet3\tnet4\n")
        outf.flush()

        for n in range(rule_size):
            concordancia_total = 0

            spectrum_classes = [0] * 4
            network_classes = [0] * 4

            print("Compute rule", n)

            for ic in ics:
                te = ca.cellularautomata(n, k, r, ca.from_number_fix(ic, k, t), t, transient)
                
                spectrum_classified = nspec.spectrum_predict_classification(te, k, r, db)
                network_classified = network_predict_classification(te, network_estimator)

                spectrum_classes[spectrum_classified - 1] += 1
                network_classes[network_classified - 1] += 1

                if spectrum_classified == network_classified:
                    concordancia_total += 1

            total = float(len(ics))

            spectrum_classes = [float(s)/total for s in spectrum_classes]
            network_classes = [float(net)/total for net in network_classes]

            print("Write resulte for rule", n)
            outf.write(str(n))
            outf.write("\t")
            outf.write(str(float(concordancia_total)/float(len(ics))))
            outf.write("\t")
            for val in spectrum_classes:
                outf.write(str(val))
                outf.write("\t")
            for val in network_classes:
                outf.write(str(val))
                outf.write("\t")
            outf.write("\n")
            outf.flush()

###############################################################################
def main(argv):
    if len(argv) < 7:
        print("Usage: python ca_class_eval.py k, r, size, total_ics, model_path, database_file, output_file[, output_encoding]")
        return

    print("Start evaluating...")

    k = int(argv[0])
    r = float(argv[1])
    total_ics = int(argv[2])
    size = int(argv[3])
    model_path = argv[4]
    database_file = argv[5]
    output_file = argv[6]

    dataset_file = Path(database_file)

    if not dataset_file.exists():
        print("Create spectrum dataset")    
        nspec.create_spectrum_dataset(k, r, size, 3*size, 2*size, total_ics, database_file)

    if len(argv) >= 8:
        output_encoding = argv[7]
        predict_with_network_and_spectrum(k, r, total_ics, size, model_path, database_file, output_file, output_encoding)
    else:
        predict_with_network_and_spectrum(k, r, total_ics, size, model_path, database_file, output_file)

    print("Finish.")

###############################################################################
if __name__ == "__main__":
    main(sys.argv[1:])
