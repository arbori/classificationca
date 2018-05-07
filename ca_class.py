"""Convolutional Neural Network Estimator for CA Classification, built with tf.layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

import random as rnd

import cellularautomata as ca
import classificationca as cca

tf.logging.set_verbosity(tf.logging.INFO)

###############################################################################
config = {}
config["kernel_size"] = [5,5]
config["padding="] = "same"
config["steps"] = 1000
config["batch_size"] = 200

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
        kernel_size = config["kernel_size"], 
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
        kernel_size = config["kernel_size"], 
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
        kernel_size=config["kernel_size"], 
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
def network_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # Temporal Evolution are 100x100 cells, and have one color channel (black or white)
    input_layer = tf.reshape(features["x"], [-1, 100, 100, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 3x3 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 100, 100, 1]
    # Output Tensor Shape: [batch_size, 100, 100, 4]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=4, 
        kernel_size = config["kernel_size"], 
        padding="same", 
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 100, 100, 4]
    # Output Tensor Shape: [batch_size, 50, 50, 4]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 3x3 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 50, 50, 4]
    # Output Tensor Shape: [batch_size, 50, 50, 8]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=8, 
        kernel_size = config["kernel_size"], 
        padding="same", 
        activation=tf.nn.relu)

    # Pooling Layer #2
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 50, 50, 8]
    # Output Tensor Shape: [batch_size, 25, 25, 8]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Convolutional Layer #3
    # Computes 128 features using a 3x3 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 25, 25, 8]
    # Output Tensor Shape: [batch_size, 25, 25, 16]
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=16,
        kernel_size=config["kernel_size"], 
        padding="same", 
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 25, 25, 16]
    # Output Tensor Shape: [batch_size, 12, 12, 16]
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 12, 12, 16]
    # Output Tensor Shape: [batch_size, 12 * 12 * 16]
    pool3_flat = tf.reshape(pool3, [-1, 12 * 12 * 16])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 12 * 12 * 16]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.7 probability that element will be kept
    ### Why drouput elements?
        #==> Dropout: A Simple Way to Prevent Neural Networks from Overfitting (pdf)
        ### So, how define the rate?
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.7, training=mode == tf.estimator.ModeKeys.TRAIN)

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
def train_function(k, r, t, transient, train_data_size, train_file, eval_file, model_diretory):
    eval_data_size = int(0.1*train_data_size)
    if eval_data_size <= 0:
        eval_data_size = 1

    train_data_file = Path(train_file)
    eval_data_file = Path(eval_file)

    train_data = []
    train_labels = []
    eval_data = []
    eval_labels = []

    # Generate/Load training and eval data
    if (not train_data_file.exists()) or (not eval_data_file.exists()):
        print("Generate training data")
        train_array = cca.make_set_pair_evolution_label(train_data_size, k, r, t, transient)
        print("Generate eval data")
        eval_array = cca.make_set_pair_evolution_label(eval_data_size, k, r, t, transient)

        print("Save training data")
        np.save(train_data_file, train_array)
        print("Save eval data")
        np.save(eval_data_file, eval_array)
    else:
        print("Load training data")
        train_array = []
        train_data = np.load(train_data_file)

        train_array.append([line for line in train_data[0]])
        train_array.append([line for line in train_data[1]])
        
        print("Load eval data")
        eval_array = []
        eval_data = np.load(eval_data_file)

        eval_array.append([line for line in eval_data[0]])
        eval_array.append([line for line in eval_data[1]])

    # Set training and eval data
    print("Set data to train/eval the network")
    train_data = np.array(train_array[0], dtype=np.float32)
    train_labels = np.array(train_array[1], dtype=np.int32)

    eval_data = np.array(eval_array[0], dtype=np.float32)
    eval_labels = np.array(eval_array[1], dtype=np.int32)

    # Create the Estimator
    print("Create the Estimator")
    ca_classifier = tf.estimator.Estimator(
        #model_fn=cnn_model_fn, model_dir=model_diretory)
        model_fn=network_model_fn, model_dir=model_diretory)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    print("Set log")
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    print("Train the model")
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=config["batch_size"],
        num_epochs=None,
        shuffle=True)
    ca_classifier.train(
        input_fn=train_input_fn,
        steps=config["steps"],
        hooks=[logging_hook])

    # Evaluate the model and print results
    print("Evaluate the model and print results")
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    eval_results = ca_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

###############################################################################
def main(argv):
    if len(argv) < 8:
        print("Usage: python ca_class.py k r t transient train_data_size train_file eval_file model_diretory")
        return

    print("Start training...")

    k = int(argv[0])
    r = float(argv[1])
    t = int(argv[2])
    transient = int(argv[3])
    train_data_size = int(argv[4])
    train_file = argv[5]
    eval_file = argv[6]
    model_diretory = argv[7]

    train_function(k, r, t, transient, train_data_size, train_file, eval_file, model_diretory)

###############################################################################
if __name__ == "__main__":
    main(sys.argv[1:])
