print("Importing modules...")

#import os
#os.environ.setdefault('PATH', '')
import time
import datetime

#import sys
#from pathlib import Path
#import tensorflow as tf
import numpy as np
import random as rd

import cellularautomata as ca
import classificationca as cca

###############################################################################
def space_spectrum(k, r, length, t, transient):
    m = 2*r + 1

    rule_size = int(pow(k, pow(k, m)))
    ic_size = pow(k, length)

    neigh_freq = [0.0] * ic_size

    begin = time.time()
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

    print(st, " - Begin")

    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    print(st, " - Initial Conditons:", ic_size)

    with open('space_spectrum.out.txt', mode='w', encoding='utf-8') as a_file:
        for n in range(rule_size):
            for ic_num in range(ic_size):
                ic = cca.from_number_fix(ic_num, k, length)

                te = ca.cellularautomata(n, k, r, ic, t, transient)

                begin = time.time()

                for line in range(len(te)):
                    for cell in range(len(te[0])):
                        winnum = cca.winNumber(te, cell, line, int(2*r+1), int(2*r+1), k)
                        neigh_freq[winnum] += 1

                if ic_num % 1000 == 0:
                    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
                    print(st, " - Check point", ic_num, ":", int(time.time() - begin), "\bs")

            st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            print(st, " - Salvando spectro da regra ", n, "...")

            a_file.write(str(n))
            a_file.write("\t")
            a_file.write(str(cca.WolframClassification[n]))
            a_file.write("\t")
            total = np.sum(neigh_freq)
            for val in neigh_freq:
                a_file.write(str(val/total))
                a_file.write("\t")
            a_file.write("\n")

    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    print(st, " - Done in", int(time.time() - begin), "\bs")
    print(st, " - End")

#------------------------------------------------------------------------------
if __name__ == '__main__':
  k = 2
  r = 1
  length = 15
  t = 3*length
  transient = 2*length
  space_spectrum(k, r, length, t, transient)