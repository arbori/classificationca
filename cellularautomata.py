import random as rnd

###############################################################################
def plain(env):
    array = []
    
    l = 0
    L = len(env)

    while l < L:
        array = array + env[l]

        l += 1
    
    return array

###############################################################################
def split(arr, size):
    arrs = []
    
    while len(arr) > size:
        pice = arr[:size]
        arrs.append(pice)
        arr   = arr[size:]
    
    arrs.append(arr)
    
    return arrs

###############################################################################
def tree_size(k, r):
    sizeof = int(k);
    m = int(2 * r + 1);

    size = 0;
    for x in range(m):
        size += sizeof;

        sizeof *= 2;

    return size

###############################################################################
def tree_fill(n, k, r) :
    size = tree_size(k, r)
    tree = [0] * size
    i = len(tree) - int(pow(k, 2 * r + 1))

    # Decompõe a regra em digitos binários, guardando-os nas folhas da arvore.
    while i < size:
        tree[i] = int(n % k)
        n = n / k

        i += 1
    
    return tree

###############################################################################
def transition_function(tree, env, r, length, i):
    pointer = int(i - r) if i >= r else int(length - r + i)

    node = -1;

    m = int(2 * r + 1)
    while m > 0:
        node = 2 * node + 2 + env[pointer];
        pointer += 1

        if pointer >= length :
            pointer -= length
        
        m -= 1

    return tree[node]

###############################################################################
def apply_rule(tar, src, ruleTree, X, r):
    for x in range(X) :
        tar[x] = transition_function(ruleTree, src, r, X, x)

###############################################################################
def cellularautomata(n, k, r, ic, t, transient=0):
    # Define o tamanho do reticulado
    width = len(ic)
    # O tamanho do resultado leva em conta a evolução temporal mais uma linha
    # para a condição inicial.
    height = int(t - transient);

    # Cria a evolução temporal e copia a condição inicial
    te = [0] * (height * width)
    te = split(te, width)
    for i in range(width) : te[0][i] = ic[i]

    # Constroi a arvore de decisão para a regra.
    ruleTree = tree_fill(n, k, r)

    # Para cada momento da do período transiente, ...
    for y in range(transient) :
        # ... se a momento for par, guarda o resultado na segunda linha.
        if (y % 2 == 0) :
            apply_rule(te[1], te[0], ruleTree, width, r)
        # Se a momento for impar, guarda o resultado na primeira linha.
        else :
            apply_rule(te[0], te[1], ruleTree, width, r)

    # Troca os valores da primeira linha pelos das segunda para que
    # a última evolução temporal esteja na primeira linha.
    if (transient % 2 != 0) :
        apply_rule(te[0], te[1], ruleTree, width, r)

    # Após o período transiente, processa a evolução temporal.
    for y in range(1, height) :
        apply_rule(te[y], te[y - 1], ruleTree, width, r);

    return te

#------------------------------------------------------------------------------
def entropy_evolution_elementar_space():

    for classe in WolframClasses:
        class_number += 1

        print(date_log()+': Compute class '+str(class_number)+'...')

        for n in classe:
            print(date_log()+': Compute rule '+str(n)+'...')

            ees = []
            X = []

            print(date_log()+': Compute entropy...')
            mean_total = 0
            while mean_total < mean_length:
                ees.append(retrieve_entropy_evolution(n, k, r, width, T, transient)[0])

                mean_total += 1

            ees = np.transpose(ees)

            print(date_log()+': Compute average and standard deviation...')
            for line in ees:
                X.append(np.average(line))
                X.append(np.std(line))

            X = ca.split(X, 2)

            print(date_log()+': Write file for class '+str(class_number)+', rule '+str(n)+'...')
            entropyf = open('C:/Users/arbori/OneDrive/workspacePhyton/classificationca/output/entropy_class_'+str(class_number)+'_n'+str(n)+'.txt','w+')
            for x in X:
                for value in x:
                    entropyf.write(str(value))
                    entropyf.write('    ')
                entropyf.write('\n')

            entropyf.close()

    print(date_log()+': Finish.')

###############################################################################
'''
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

#------------------------------------------------------------------------------
def show_graph(spectrum, width, height):
  mat = spectrum[0:height]
  filtered = []
  for lin in mat:
    filtered.append(lin[0:width])

  z = np.array(filtered)

  # make these smaller to increase the resolution
  dx, dy = 1.0, 1.0

  # generate 2 2d grids for the x & y bounds
  y, x = np.mgrid[slice(1, len(z) + dy, dy),
                  slice(1, len(z) + dx, dx)]

  # x and y are bounds, so z should be the value *inside* those bounds.
  # Therefore, remove the last value from the z array.
  z = z[:-1, :-1]
  levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())

  # pick the desired colormap, sensible levels, and define a normalization
  # instance which takes data values and translates those into levels.
  cmap = plt.get_cmap('gray_r')
  norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

  fig, (ax1) = plt.subplots(nrows=1)

  # contours are *point* based plots, so convert our bound into point
  # centers
  cf = ax1.contourf(x[:-1, :-1] + dx/2.,
                    y[:-1, :-1] + dy/2., z, levels=levels,
                    cmap=cmap)
  fig.colorbar(cf, ax=ax1)
  ax1.set_title('contourf with levels')

  # adjust spacing between subplots so `ax1` title and `ax0` tick labels
  # don't overlap
  fig.tight_layout()

  plt.show()

#------------------------------------------------------------------------------
n = 30531
k = 2
r = 1.5
length = 100
t = 3*length
transient = 2*length
ic = [rnd.randrange(0,k) for _ in range(length)]

te = cellularautomata(n, k, r, ic, t, transient)
'''