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

import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

import random as rnd

import cellularautomata as ca
import classificationca as cca
import ca_class as cac
import neigbor_spectrum as nspec

tf.logging.set_verbosity(tf.logging.INFO)

###############################################################################
def classify_space_rule(k, r, size, total_ics, model_path, output_file, output_encoding='utf-8'):
    t = 3*size
    transient = 2*size

    m = 2*r + 1
    rule_size = int(pow(k, pow(k, m)))

    # Create the Estimator
    print("Create the Estimator")
    ca_classifier = tf.estimator.Estimator(
        #model_fn=cac.cnn_model_fn, model_dir=model_path)
        model_fn=cac.network_model_fn, model_dir=model_path)

    # Predict CA in ray 1 space
    with open(output_file, mode='w', encoding=output_encoding) as a_file:
        for n in range(rule_size):
            print("Rule:", n)

            eval_data = cca.make_temporal_evolutions_rule(total_ics, n, k, r, t, transient)

            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": np.array(eval_data[0], dtype=np.float32)},
                num_epochs=1,
                shuffle=False)

            result_predict = ca_classifier.predict(eval_input_fn, predict_keys=None, hooks=None)

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
        model_fn=cac.cnn_model_fn, model_dir=model_path)

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
                
                network_classified = network_predict_classification(te, network_estimator)
                spectrum_classified = nspec.spectrum_predict_classification(te, k, r, db)

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
        print("Usage: python ca_class_eval.py process k r size total_ics model_path database_file output_file [output_encoding]")
        print("\tprocess == \"classify_space_rule\" or process == \"predict_with_network_and_spectrum\"")
        return

    print("Start evaluating...")

    process = argv[0]
    k = int(argv[1])
    r = float(argv[2])
    total_ics = int(argv[3])
    size = int(argv[4])
    model_path = argv[5]
    database_file = argv[6]
    output_file = argv[7]
    output_encoding = "utf-8"

    if len(argv) > 8:
        output_encoding = argv[8]

    dataset_file = Path(database_file)

    print("Parameters")
    print("\tprocess:", process)
    print("\tk:", k)
    print("\tr:", r)
    print("\ttotal_ics:", total_ics)
    print("\tsize:", size)
    print("\tmodel_path:", model_path)
    print("\tdatabase_file:", database_file)
    print("\toutput_file:", output_file)
    print("\toutput_encoding:", output_encoding)
    print()

    if process == "classify_space_rule":
        classify_space_rule(k, r, size, total_ics, model_path, output_file, output_encoding)

    elif process == "predict_with_network_and_spectrum":
        if not dataset_file.exists():
            print("Create spectrum dataset")    
            nspec.create_spectrum_dataset(k, r, size, 3*size, 2*size, total_ics, database_file, output_encoding)

        if len(argv) >= 9:
            output_encoding = argv[7]
            predict_with_network_and_spectrum(k, r, total_ics, size, model_path, database_file, output_file, output_encoding)
        else:
            predict_with_network_and_spectrum(k, r, total_ics, size, model_path, database_file, output_file)
    else:
        print("Process choosed was not defined.")
        print("process == \"classify_space_rule\" or process == \"predict_with_network_and_spectrum\"")
        print()

    print("Finish.")

###############################################################################
if __name__ == "__main__":
    main(sys.argv[1:])
