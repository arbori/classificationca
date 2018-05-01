import random as rnd
import numpy as np

###############################################################################
def from_number(number, base):
    digits = []

    if number < base:
        return [number]

    while number >= base:
        digits.append(number % base)
        number = int(number/base)

    digits.append(number)

    return list(digits)

###############################################################################
def from_number_fix(number, base, size):
    digits = from_number(number, base)

    if len(digits) > size:
        return digits[:size]

    while len(digits) < size:
        digits = digits + [0]

    return digits

###############################################################################
def from_digits(base, digits):
    """Convert from base b to a positive decimal number the recieved digit list

    Keyword arguments:
    base -- the base of the digits is
    digits -- list of digits of the number in base b
    """
    if base <= 0 or len(digits) <= 0:
        return -1

    return round(
        np.sum(
            np.multiply(
                digits, 
                np.power(
                    base,
                    list(range(len(digits)))
                )
            )
        )
    )

###############################################################################
def plain(env):
    array = []
    
    for line in env:
        array = array + line
    
    return array

###############################################################################
def split(arr, size):
    arrs = []
    
    while len(arr) > size:
        pice = arr[:size]
        arrs.append(pice)
        arr  = arr[size:]
    
    arrs.append(arr)
    
    return arrs

###############################################################################
def get_neighborhood(i, r, line):
    result = []

    ini = int(i - r)

    if ini < 0:
        ini += len(line)

    for _ in range(int(2*r + 1)):
        result.append(line[ini])

        ini += 1
        if ini >= len(line):
            ini -= len(line)
        
    return result

###############################################################################
def create_rule(n, k, r) :
    '''
    Create transition function (rule) based on rule number n, possibles states k
    and size of the ray r.
    '''
    m = int(2*r + 1) # Neighborhood size
    Y = from_number_fix(n, k, pow(k, m)) # Neighborhood transitions

    transitions = [[from_number_fix(y, k, m), Y[y]] for y in range(len(Y))]
    
    return transitions

###############################################################################
def transition_function(rule, line, r, i):
    '''
    Aplly rule definition to the cell gived in line parameter in i position
    '''
    neighborhood = get_neighborhood(i, r, line)

    for transition in rule:
        if neighborhood == transition[0]:
            return transition[1]
    
    return -1

###############################################################################
def apply_rule(tar, src, rule, r):
    '''
    Aplly rule definition under environment tar based on configuration 
    of src.
    '''
    width = len(src)

    for i in range(width) :
        tar[i] = transition_function(rule, src, r, i)

###############################################################################
def cellularautomata(n, k, r, ic, t, transient=0):
    '''
    Generates a list, that each cell may be one of k possible states,
    representing the evolution of the cellular automaton with the specified 
    rule n, the ray value equal to r, from initial condition ic, for t - transient 
    steps.
    '''
    # Define the width of the reticulate based on initial condition wodth
    width = len(ic)
    # Compute the number of evolutions steps.
    height = int(t - transient)

    # Make the temporal evolution and copy initial condition
    te = [0] * (height * width)
    te = split(te, width)
    for i in range(width) : te[0][i] = ic[i]

    # Constroi a arvore de decisão para a regra.
    rule = create_rule(n, k, r)

    # Para cada momento da do período transiente, ...
    for y in range(transient) :
        # ... se a momento for par, guarda o resultado na segunda linha.
        if (y % 2 == 0) :
            apply_rule(te[1], te[0], rule, r)
        # Se a momento for impar, guarda o resultado na primeira linha.
        else :
            apply_rule(te[0], te[1], rule, r)

    # Troca os valores da primeira linha pelos das segunda para que
    # a última evolução temporal esteja na primeira linha.
    if (transient % 2 != 0) :
        apply_rule(te[0], te[1], rule, r)

    # Após o período transiente, processa a evolução temporal.
    for y in range(1, height) :
        apply_rule(te[y], te[y - 1], rule, r)

    return te

'''
#------------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator

    k = 2
    r = 1.5
    rule_size = int(pow(k, pow(k, 2*r + 1)))

    n = 1142 #rnd.randint(0, rule_size)
    length = 50
    t = 3*length
    transient = 2*length
    ic = [rnd.randrange(0,k) for _ in range(length)]

    te = cellularautomata(n, k, r, ic, t, transient)

    title = "Temporal Evolution - Rule " + str(n)

    plt.style.use('grayscale')
    fig, (ax) = plt.subplots(ncols=1)
    fig.suptitle(title)

    ax.imshow(te, interpolation='none')

    plt.show()
'''