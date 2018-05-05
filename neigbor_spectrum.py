print("Importing modules...")

#import os
#os.environ.setdefault('PATH', '')
import time
import datetime
from pathlib import Path
import sys

#import sys
#from pathlib import Path
import numpy as np
import random as rd
import csv

import cellularautomata as ca
import classificationca as cca

###############################################################################
def get_tmstr():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

###############################################################################
def neighborhood_configuration_spectrum(te, k, r):
    '''
    Create the temporal evolution's spectrum from temporal evolution te, 
    generated with k states and ray with value r.
    '''
    spectrum = [0.0] * int(pow(k, (2*r + 1)*(2*r + 1)))

    for line in range(len(te)):
        for cell in range(len(te[0])):
            index = cca.winNumber(te, cell, line, int(2*r+1), int(2*r+1), k)

            if index >= len(spectrum):
                print(te, index, cell, line, int(2*r+1), int(2*r+1), k)

            spectrum[index] += 1

    return spectrum

###############################################################################
def create_spectrum_dataset(k, r, length, t, transient, experiments, dataset_file, encoding_type='utf-8'):
    '''
    It create a elementar space's spectro dataset 
    '''
    m = int(2*r + 1)
    rule_size = int(pow(k, pow(k, m)))
    neighborhood_size = int(pow(k, m*m))
    spectrum_result = [0] * neighborhood_size

    print(get_tmstr(), " - Begin")
    print(get_tmstr(), " - Rule size:", rule_size)

    print("Abrindo o arquivo", dataset_file)

    with open(dataset_file, mode='w', encoding=encoding_type) as a_file:
        for n in range(rule_size):
            begin = time.time()
            
            print(get_tmstr(), " - Rule", n)

            for i in range(experiments):
                ic = ca.from_number_fix(rd.randint(0, int(pow(k, length))), k, length)

                te = ca.cellularautomata(n, k, r, ic, t, transient)
                spectrum = neighborhood_configuration_spectrum(te, k, r)
                spectrum_result = [spectrum_result[s] + spectrum[s] for s in range(len(spectrum_result))]

                if i % int(experiments / 10) == 0:
                    print(get_tmstr(), " - Check point", i, ":", int(time.time() - begin), "\bs")

            spectrum_result = cca.normalize(spectrum_result)

            print(get_tmstr(), " - Salvando spectro da regra ", n, "...")

            a_file.write(str(n))
            a_file.write("\t")
            a_file.write(str(cca.WolframClassification[n]))
            a_file.write("\t")
            for val in spectrum_result:
                a_file.write(str(val))
                a_file.write("\t")
            a_file.write("\n")
            a_file.flush()

    print(get_tmstr(), " - Done in", int(time.time() - begin), "\bs")
    print(get_tmstr(), " - End")

###############################################################################
def retrieve_spectrum_dataset(dataset_file):
    '''
    Retrieve from dataset file, the elementar space's spectrum dataset.
    '''
    dataset = []

    with open(dataset_file, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='	', quotechar='"')

        for row in spamreader:
            line = []
    
            for i in range(len(row)):
                if i == 0 or i == 1:
                    line.append(int(row[i]))
                else:
                    if row[i] != "":
                        line.append(float(row[i]))

            dataset.append(line)

    return dataset

###############################################################################
def spectrum_predict_classification(te, k, r, db):
    '''Predict the temporal evolution class (te) using a spectrum database (db).
    The temporal evolution must be build with k states and neighborhood ray r. 
    '''
    predict = 0
    minimum = 10e100
    spectrum = cca.normalize(neighborhood_configuration_spectrum(te, k, 1.0))

    for row in range(len(db)):
        s = db[row][2:]

        d = np.sum([(s[i]-spectrum[i])*(s[i]-spectrum[i]) for i in range(len(s))])

        if minimum > d:
            predict = row
            minimum = d
        
        if minimum == 0.0:
            break

    if predict >= 0:
        return db[predict][1]
    
    return predict

###############################################################################
def classify_elementar_space_spectro(k, r, length, t, transient, experiments, db_name, output_file_name=None):
    '''
    It use the elementar space's spectral database to create a classification to
    space (k,r) and return the classification table 
    '''
    classification = []

    rule_size = int(pow(k, pow(k, 2*r + 1)))

    db = retrieve_spectrum_dataset(db_name)
    fout = None

    if output_file_name != None:
        fout = open(output_file_name, mode='w', encoding='utf-8')

    for n in range(rule_size):
        right_classes = 0

        for _ in range(experiments):
            ic = ca.from_number_fix(rd.randint(0, int(pow(k, length))), k, length)
            te = ca.cellularautomata(n, k, r, ic, t, transient)

            WolframClass = cca.WolframClassification[n]
            spectrumClass = spectrum_predict_classification(te, k, r, db)

            if WolframClass == spectrumClass:
                right_classes += 1
            
        classification.append([
            n, 
            cca.WolframClassification[n], 
            float(right_classes)/float(experiments)])
        
        if output_file_name != None:
            fout.write(str(n))
            fout.write("\t")
            fout.write(str(cca.WolframClassification[n]))
            fout.write("\t")
            fout.write(str(float(right_classes)/float(experiments)))
            fout.write("\n")

            fout.flush()

    fout.close()

    return classification

###############################################################################
###############################################################################
###############################################################################
def main(argv):
    #if len(argv) < 7:
    #    print("Usage: python neigbor_spectrum.py k r length t transient db_name experiments [output_file_name]")
    #    return

    print("Start classifing...")

    k = int(argv[0])
    r = float(argv[1])
    length = int(argv[2])
    t = int(argv[3])
    transient = int(argv[4])
    db_name = argv[5]
    experiments = int(argv[6])
    output_file_name = "neigbor_spectrum.csv"

    if len(argv) >= 8:
        output_file_name = argv[7]

    print(
        "Classifing rule space with", k, "states and radius", r, "\n\t"
        "Length:", length, "units\n\t",
        "Time:", t, "times steps\n\t",
        "Transients:", transient, "steps\n\t",
        "Database file:", db_name, "\n\t",
        "Experiments:", experiments, "\n\t",
        "Output file:", output_file_name
    )

    dataset_file = Path(db_name)

    if not dataset_file.exists():
        print("Create spectrum dataset")
        create_spectrum_dataset(k, r, length, t, transient, experiments, db_name)

    print("Classifing space r", r)
    classification = classify_elementar_space_spectro(k, r, length, t, transient, experiments, db_name, output_file_name)

    print("It was", len(classification), "rules classifiede.")
    print("Finish.")

###############################################################################
if __name__ == "__main__":
    main(sys.argv[1:])
