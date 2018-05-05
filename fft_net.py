import os
os.environ.setdefault('PATH', '')
import time

import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
import random as rd

import cellularautomata as ca
import classificationca as cca

###############################################################################
## ...
checkpoint_prefix = "./prediction"
samples_per_class = 256
epoch_max = 60

error_threshold = 10e-3

## Define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [100, 100])
ys = tf.placeholder(tf.float32, [4])

## Cellular Automata parameters
k = 2
r = 1
length = 100
t = 3*length
transient = 2*length
rule_size = int(np.power(k, np.power(k, 2*r + 1)))

#------------------------------------------------------------------------------
def make_prediction(xs):
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
def get_rule(clas=None):
    if clas == None or clas < 0 or clas > 3:
        clas = rd.randrange(0,3)

    n = cca.WolframClasses[clas][rd.randrange(0,len(cca.WolframClasses))]

    return n

###############################################################################
def get_data(n, ic=None, apply_subclass=False):
    if apply_subclass:
        n = cca.get_subclass(n)

    if ic == None:
        ic = [rd.randrange(0,k) for _ in range(length)]

    te = ca.cellularautomata(n, k, r, ic, t, transient)

    x_result = cca.power_spectrum(cca.get_fft(te))
    y_result = [0 for _ in range(4)]
    y_result[cca.wolfram_class(n) - 1] = 1

    # Get data
    return x_result, y_result

###############################################################################
def training_net(sess, prediction):
    #--------------------------------------------------------------------------
    ## The error between prediction and real data
    #--------------------------------------------------------------------------
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction)))

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    #--------------------------------------------------------------------------
    print("Training rule...")

    ics = cca.init_conditions_numbers(k, length, samples_per_class)
    #samples = range(samples_per_class)

    for n in range(0,255):
        localtime = time.asctime( time.localtime(time.time()) )
        print(localtime, ":", "Treinando regra", n)

        error = 0.0
        total_train_step = 0
    
        for ic in ics:
            icbin = ca.from_number_fix(ic, 2, length)

            erroic = 10e9
            #tempted = 0

        while erroic > error_threshold:
            # Get data
            x_data, y_data = get_data(n, icbin)

            ## Training
            sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
            ## Update the loost mean                
            erroic = sess.run(loss, feed_dict={xs:x_data, ys:y_data})

        total_train_step += 1
        error += erroic

    localtime = time.asctime( time.localtime(time.time()) )
    print(localtime, ":", n, (error/total_train_step))

    return prediction

###############################################################################
def compute_average_acuracy(sess, prediction):
    print("It is computing average acuracy...")

    error = 0

    for _ in range(int(0.25*epoch_max)):
        # Get data
        n = get_rule()
        x_data, y_data = get_data(n)

        y = sess.run(prediction, feed_dict={xs:x_data, ys:y_data})
        y = [int(round(i)) for i in y]

        if y != y_data:
            error += 1

        if np.sum(y) != 1:
            print("Resposta inconsistente:", n, y)

    predict_average = (1.0 - (error/(0.25*epoch_max)))

    return predict_average

###############################################################################
def predict_class(sess, n, k, r, prediction):
    rules = np.array([0,0,0,0])

    for _ in range(int(0.10*epoch_max)):
        # Get transformed temporl evolution
        x_data, y_data = get_data(n)

        y = sess.run(prediction, feed_dict={xs:x_data, ys:y_data})
        y = np.array([int(round(i)) for i in y])
        rules = rules + y

    total = np.sum(rules)
    rules = [rl/total for rl in rules]

    return rules

#------------------------------------------------------------------------------
if __name__ == '__main__':
    ## Tensorflow session
    with tf.Session() as sess:
        predict_average = 0.0

        prediction = make_prediction(xs)

        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver()

        there_is_checkpoint = tf.train.checkpoint_exists(checkpoint_prefix)

        if there_is_checkpoint:
            saver.restore(sess, checkpoint_prefix)
            predict_average = 1.0
        else:
            prediction = training_net(sess, prediction)
            predict_average = compute_average_acuracy(sess, prediction)

        print("Average acuracy:", predict_average)

    print("Saving Neural Network...")
    save_path = saver.save(sess, checkpoint_prefix)

    print("Predicting class...")
    print("States: ", k, " Ray: ", r)

    with open(checkpoint_prefix + ".txt", "w+") as f:
        n = 0
        k = 2
        r = 1
        rule_size = int(np.power(k, np.power(k, 2*r + 1)))

        for n in range(rule_size):
            prob = predict_class(sess, n, k, r, prediction)

            print(n, "/", rule_size, cca.wolfram_class(n), prob)

            f.write(str(n))
            f.write(str('\t'))
            f.write(str(cca.wolfram_class(n)))
            for p in prob:
                f.write(str('\t'))
                f.write(str(p))
            f.write(str('\n'))

            if n % 10 == 0:
                f.flush()

    print()
    print("Finish.")
