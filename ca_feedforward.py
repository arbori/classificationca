"""Convolutional Neural Network Estimator for CA Classification, built with tf.layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from pathlib import Path

import random as rnd

import cellularautomata as ca
import classificationca as cca

tf.logging.set_verbosity(tf.logging.INFO)

###############################################################################
def feedforward_model(features, labels, mode):
    W1 = tf.Variable(tf.random_normal([100,100]))
    b1 = tf.Variable(tf.zeros([100]) + 0.1)
    Wx1_plus_b = tf.matmul(xs, W1) + b1
    layer1 = tf.nn.sigmoid(Wx1_plus_b) # 100x100

    W2 = tf.Variable(tf.random_normal([100,90]))
    b2 = tf.Variable(tf.zeros([100,1]) + 0.1)
    Wx2_plus_b = tf.matmul(layer1, W2) + b2 # 100x90
    layer2 = tf.transpose(tf.nn.sigmoid(Wx2_plus_b)) # 90x100

    W3 = tf.Variable(tf.random_normal([100,80]))
    b3 = tf.Variable(tf.zeros([90,1]) + 0.1)
    Wx3_plus_b = tf.matmul(layer2, W3) + b3 # 90x80
    layer3 = tf.transpose(tf.nn.sigmoid(Wx3_plus_b)) # 80x90

    W4 = tf.Variable(tf.random_normal([90, 70]))
    b4 = tf.Variable(tf.zeros([80, 1]) + 0.1)
    Wx4_plus_b = tf.matmul(layer3, W4) + b4 # 80x70
    layer4 = tf.transpose(tf.nn.sigmoid(Wx4_plus_b)) # 70x80

    W5 = tf.Variable(tf.random_normal([80, 60]))
    b5 = tf.Variable(tf.zeros([70, 1]) + 0.1)
    Wx5_plus_b = tf.matmul(layer4, W5) + b5 # 70x60
    layer5 = tf.transpose(tf.nn.sigmoid(Wx5_plus_b)) # 60x70

    W6 = tf.Variable(tf.random_normal([70, 50]))
    b6 = tf.Variable(tf.zeros([60, 1]) + 0.1)
    Wx6_plus_b = tf.matmul(layer5, W6) + b6 # 60x50
    layer6 = tf.transpose(tf.nn.sigmoid(Wx6_plus_b)) # 50x60

    W7 = tf.Variable(tf.random_normal([60, 40]))
    b7 = tf.Variable(tf.zeros([50, 1]) + 0.1)
    Wx7_plus_b = tf.matmul(layer6, W7) + b7 # 50x40
    layer7 = tf.transpose(tf.nn.sigmoid(Wx7_plus_b)) # 40x50

    W8 = tf.Variable(tf.random_normal([50, 30]))
    b8 = tf.Variable(tf.zeros([40, 1]) + 0.1)
    Wx8_plus_b = tf.matmul(layer7, W8) + b8 # 40x30
    layer8 = tf.transpose(tf.nn.sigmoid(Wx8_plus_b)) # 30x40

    W9 = tf.Variable(tf.random_normal([40, 20]))
    b9 = tf.Variable(tf.zeros([30, 1]) + 0.1)
    Wx9_plus_b = tf.matmul(layer8, W9) + b9 # 30x20
    layer9 = tf.transpose(tf.nn.sigmoid(Wx9_plus_b)) # 20x30

    W10 = tf.Variable(tf.random_normal([30, 10]))
    b10 = tf.Variable(tf.zeros([20, 1]) + 0.1)
    Wx10_plus_b = tf.matmul(layer9, W10) + b10 # 20x10
    layer10 = tf.transpose(tf.nn.sigmoid(Wx10_plus_b)) # 10x20

    W11 = tf.Variable(tf.random_normal([20, 4]))
    b11 = tf.Variable(tf.zeros([10, 1]) + 0.1)
    Wx11_plus_b = tf.matmul(layer10, W11) + b11 # 10x4
    layer11 = tf.transpose(tf.nn.sigmoid(Wx11_plus_b)) # 4x10

    W12 = tf.Variable(tf.random_normal([10, 1]))
    b12 = tf.Variable(tf.zeros([4, 1]) + 0.1)
    Wx12_plus_b = tf.reshape(tf.matmul(layer11, W12) + b12, [4])
    predict = tf.nn.sigmoid(Wx12_plus_b)

    return predict

###############################################################################
def main(unused_argv):
    # Define parameters of cellular automata
    k = 2
    r = 1
    t = 100

    train_data_size = 400
    eval_data_size = int(0.1*train_data_size)
    if eval_data_size <= 0:
        eval_data_size = 1

    train_data_file = Path("./ca_feedforward.train.data.npy")
    eval_data_file = Path("./ca_feedforward.eval.data.npy")

    train_data = []
    train_labels = []
    eval_data = []
    eval_labels = []

    # Generate/Load training and eval data
    if (not train_data_file.exists()) or (not eval_data_file.exists()):
        print("Generate training data")
        train_array = cca.make_set_pair_evolution_label(train_data_size, k, r, 3*t, 2*t)
        print("Generate eval data")
        eval_array = cca.make_set_pair_evolution_label(eval_data_size, k, r, 3*t, 2*t)

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
    mnist_classifier = tf.estimator.Estimator(
        model_fn=feedforward_model, model_dir="./ca_feedforward_classification")

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
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=200,
        hooks=[logging_hook])

    # Evaluate the model and print results
    print("Evaluate the model and print results")
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

###############################################################################
if __name__ == "__main__":
    tf.app.run()
