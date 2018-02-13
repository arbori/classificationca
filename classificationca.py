import numpy as np
import random as rd
import time as tm
import datetime

import random as rnd
import cellularautomata as ca

rd.seed(int(1000*tm.time()))

WolframSubclass = [
    ### Class I ###
    [0, 234],

    ### Class II ###
    [1, 2, 3, 4, 5, 7, 9, 11, 14, 16, 17, 25, 26, 27, 28, 29, 31, 167, 33, 34, 35, 37, 39, 49, 50, 53, 57, 59, 61, 65, 67, 73, 81, 82, 83, 87, 92, 95, 108, 111, 115, 118, 123, 125, 131, 133, 142, 145, 166, 178, 209, 214, 226],

    ### Class III ###
    [18, 30, 45, 60, 75, 89, 90, 101, 102, 105, 129, 153, 165, 183, 195],

    ### class IV
    [41, 54, 97, 106, 107, 110, 120, 121, 124, 137, 147, 169, 193, 225]
]

WolframDinamicEquivalentClasses = [
    # Class I
    [0, 8, 32, 40, 128, 136, 160, 168],
    # Class II
    [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 19, 23,
    24, 25, 26, 27, 28, 29, 33, 34, 35, 36, 37, 38, 42,
    43, 44, 46, 50, 51, 56, 57, 58, 62, 72, 73, 74, 76,
    77, 78, 94, 104, 108, 130, 132, 134, 138, 140, 142,
    152, 154, 156, 162, 164, 170, 172, 178, 184, 200,
    204, 232],
    # class III
    [18, 22, 30, 45, 60, 90, 105, 122, 126, 146, 150],
    # Class IV
    [41, 54, 106, 110]
]

WolframClasses = [
    #ClassI
    [0, 8, 32, 40, 64, 96, 128, 136, 160, 168, 192, 224, 234, 235, 238, 239, 
    248, 249, 250, 251, 252, 253, 254, 255],
    #ClassII
    [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 23, 
    24, 25, 26, 27, 28, 29, 31, 33, 34, 35, 36, 37, 38, 39, 42, 43, 44, 46, 
    47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 61, 62, 63, 65, 66, 67, 68, 
    69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 91, 
    92, 93, 94, 95, 98, 99, 100, 103, 104, 108, 109, 111, 112, 113, 114, 115, 
    116, 117, 118, 119, 123, 125, 127, 130, 131, 132, 133, 134, 138, 139, 140, 
    141, 143, 144, 145, 148, 152, 154, 155, 156, 157, 158, 159, 162, 163, 164, 
    166, 167, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 184, 
    185, 186, 187, 188, 189, 190, 191, 194, 196, 197, 198, 199, 200, 201, 202, 
    203, 204, 205, 206, 207, 208, 209, 210, 211, 213, 214, 215, 216, 217, 218, 
    219, 220, 221, 222, 223, 226, 227, 228, 229, 230, 231, 232, 233, 236, 237, 
    240, 241, 242, 243, 244, 245, 246, 247],
    #classIII
    [18, 22, 30, 45, 60, 75, 86, 89, 90, 101, 102, 105, 122, 126, 129, 135, 
    146, 149, 150, 151, 153, 161, 165, 182, 183, 195],
    #ClassIV
    [41, 54, 97, 106, 107, 110, 120, 121, 124, 137, 147, 169, 193, 225]
]

###############################################################################
def wolfram_class(n):
    '''
    '''

    for classe in [0,1,2,3]:
        for rule in WolframClasses[classe]:
            if rule == n:
                return (classe + 1)
    
    return -1

###############################################################################
def normalize(mat):
    maximum = max(mat)

    result = []

    for val in mat:
        if maximum != 0: 
            result.append(val/maximum)
        else:
            result.append(0)

    return result

###############################################################################
def from_number(n, b):
    digits = []

    if n < b:
        return [n]

    while n >= b:
        digits.append(n % b)
        n = int(n/b)

    digits.append(n)

    return list(digits)

###############################################################################
def from_number_fix(n, b, size):
    digits = from_number(n, b)

    if len(digits) > size:
        return digits[:size]

    while len(digits) < size:
        digits = digits + [0]

    return digits

###############################################################################
def from_digits(b, digits):
    """Convert from base b to a positive decimal number the recieved digit list

    Keyword arguments:
    base -- the base of the digits is
    digits -- list of digits of the number in base b
    """
    if b <= 0 or len(digits) <= 0:
        return -1

    return round(
        np.sum(
            np.multiply(
                digits, 
                np.power(
                    b,
                    list(range(len(digits)))
                )
            )
        )
    )

###############################################################################
def init_conditions_numbers(k, length, total_ics):
  rnd.seed(tm.time())

  result = [0]
  limit = int(pow(k, length))
  delta = int(limit / total_ics)

  for ic in range(1, total_ics - 1):
    n = rnd.randrange(delta*ic - int(delta/2), delta*ic + int(delta/2))
    result.append(n)

  result.append(limit)

  return result

###############################################################################
# Sub-classificação baseada no comportamento observado após estado transiente.
def get_subclass(n):
    ### Class I ###
    # Leva tudo a zero. 
    if n == 0 or n == 8 or n == 32 or n == 40 or n == 64 or n == 96 or n == 128 or n == 136 or n == 160 or n == 168 or n == 192 or n == 224:
        return 0

    # Leva tudo a um.
    if n == 234 or n == 235 or n == 238 or n == 239 or n == 248 or n == 249 or n == 250 or n == 251 or n == 252 or n == 253 or n == 254 or n == 255:
        return 234

    ### Class II ###
    # Horizontais alternadas com verticais
    if n == 1 or n == 19 or n == 23 or n == 91 or n == 127:
        return 1

    # Diagonais para a esquerda com fundo branco
    if n == 2 or n == 6 or n == 10 or n == 38 or n == 42 or n == 46 or n == 66 or n == 74 or n == 98 or n == 130 or n == 138 or n == 162 or n == 163 or n == 170 or n == 194:
        return 2
    
    # 3 - Horizontais alternadas com diagonais pontilhadas descendo para a esquerda

    # Verticais com fundo branco
    if n == 4 or n == 12 or n == 13 or n == 36 or n == 44 or n == 68 or n == 69 or n == 70 or n == 71 or n == 72 or n == 76 or n == 77 or n == 78 or n == 79 or n == 100 or n == 104 or n == 132 or n == 164 or n == 196 or n == 197 or n == 198 or n == 199 or n == 200 or n == 201 or n == 204 or n == 205 or n == 228 or n == 232 or n == 236:
        return 4

    # 5 - Verticais com faixa preta central separando horizontais alternadas

    # Horizontais alternadas com diagonais com traços de 2 pixels descendo para a direita
    if n == 7 or n == 15 or n == 119:
        return 7

    # 9 - Periódica com estruturas mais complexas e faixas descendo para a direita

    # Horizontais alternadas com diagonais com traços de 2 pixels descendo para a esquerda
    if n == 11 or n == 21 or n == 63:
        return 11

    # Diagonais para a esquerda e diagonais para a direita
    if n == 14 or n == 43 or n == 62:
        return 14

    # Diagonais para a direita fundo branco
    if n == 16 or n == 20 or n == 24 or n == 48 or n == 56 or n == 80 or n == 84 or n == 88 or n == 103 or n == 112 or n == 116 or n == 134 or n == 144 or n == 152 or n == 173 or n == 176 or n == 177 or n == 208 or n == 212 or n == 240 or n == 241 or n == 244:
        return 16

    # 17 - Horizontais alternadas com diagonais pontilhadas descendo para a direita

    # Diagonais para a direita com estruturas mais complexas e fundo branco
    if n == 25 or n == 52:
        return 25

    # 26 - Diagonais para a esquerda, fundo branco, formada pelo Triângulo de Sierpinski
    # 27 - Horizontais alternadas, diagonais inclinação esquerda formadas por verticais deslocadas de 2 pixels
    # 28 - Verticais variadas, verticais de linha simples e verticais com linha simples com pontilhados lateral
    # 29 - Verticais simples e com pontilhado lateral entrecortando horizontais 
    # 31 - Horizontais alternadas com diagonais com traços de 3 pixels descendo para a direita
    # 167 - Diagonais para a esquerda, fundo preto, formada pelo Triângulo de Sierpinski

    # Horizontais alternadas verticais pontilhadas ambas deslocadas
    if n == 33 or n == 51:
        return 33

    # Diagonais para a esquerda fundo preto
    if n == 34 or n == 139 or n == 143 or n == 155 or n == 158 or n == 159 or n == 171 or n == 172 or n == 174 or n == 175:
        return 34
  
    # 35 - Diagonais distorcidas para a esquerda e diagonais para a direita formadas por linhas verticas
  
    # Verticais formadas por horizontais curtas
    if n == 37 or n == 55: 
        return 37

    # Diagonais para a direita cortadas por horizontais
    if n == 39 or n == 47:
        return 39
  
    # 49 - Diagonais pontilhadas para a direita cortadas por horizontais.
    # 50 - Tabuleiro de xadrez com verticais tracejadas
    # 53 - Diagonais para a esquerda formada por linhas verticais cortadas por horizontais
  
    # Diagonais para a esquerda pretas fundo quadriculado
    if n == 57 or n == 58 or n == 184 or n == 185 or n == 186 or n == 187 or n == 188 or n == 189 or n == 190 or n == 191:
        return 57

    # Diagonais para a direita pretas fundo quadriculado
    if n == 59 or n == 99 or n == 114 or n == 227:
        return 59
  
    # 61 - Diagonais tracejadas para a esquerda cortadas por horizontais.
    # 65 - Periódica com estruturas mais complexas e faixas descendo para a esquerda
  
    # Diagonais para a esquerda com estruturas mais complexas e fundo branco
    if n == 67 or n == 148 or n == 154:
        return 67

    # Verticais sobre estruturas complexas triangulares
    if n == 73 or n == 94:
        return 73

    # Diagonais distorcidas para a direita fundo branco
    if n == 81 or n == 85 or n == 213:
        return 81
  
    # 82 - Diagonais para a direita formada por Triângulos.
    # 83 - Diagonais para a esquerda formadas por traços.
    # 87 - Horizontais alternadas com diagonais com traços de 3 pixels descendo para a esquerda

    # Verticais com fundo preto
    if n == 92 or n == 93 or n == 156 or n == 157 or n == 202 or n == 203 or n == 206 or n == 207 or n == 216 or n == 217 or n == 218 or n == 219 or n == 220 or n == 221 or n == 222 or n == 223 or n == 233 or n == 237:
        return 92
  
    # 95 - Verticais com faixa branca central separando horizontais alternadas.

    # Verticais complexas fundo branco
    if n == 108 or n == 109 or n == 140 or n == 141:
        return 108

    # Diagonais para a direita complexas
    if n == 111 or n == 113 or n == 180 or n == 181 or n == 210 or n == 211 or n == 229:
        return 111

    # Diagonais para a direita e diagonais para a esquerda formadas por horizontais
    if n == 115 or n == 117:
        return 115

    # 118 - Diagonais para a direita e verticais complexas
    # 123 - Verticais, fundo branco, cortadas por horizontais alternadas
    # 125 - Diagonais para a esquerda, complexas II
    # 131 - Diagonais para esquerda, triâgulos verticais e poucas diagonais para direita.
    # 133 - Verticais com estruturas entre elas
    # 142 - Diagonais distorcidas para a esquerda
    # 145 - Diagonais para a direita cortadas por horizontais complexas.
    # 166 - Diagonais para a esquerda, complexas

    # Verticais sobre quadriculado
    if n == 178 or n == 179:
        return 178

    # Diagonais para a direita fundo preto
    if n == 209 or n == 215 or n == 230 or n == 231 or n == 242 or n == 243 or n == 245 or n == 246 or n == 247:
        return 209

    # 214 - Diagonais distorcidas para a direita, fundo preto
    # 226 - Diagonais para a esquerda, branca, fundo quadriculado

    ### Class III ###
    # Triângulos invertidos fundo branco
    if n == 18 or n == 22 or n == 122 or n == 126 or n == 146:
        return 18

    # Triângulos invertidos e verticais fundo branco
    if n == 30 or n == 86:
        return 30

    # 45 - Triângulos formados por verticais
    # 60 - Triângulos virados para a direita, fundo branco
    # 75 - Estrutura indefinida I para a direita
    # 89 - Estrutura indefinida I para a esquerda
    # 90 - Triângulos de Sierpinski
    # 101 - Estrutura indefinida II
    # 102 - Triângulos virados para a esquerda com fundo branco
    # 105 - Estrutura indefinida III

    # Triângulos invertidos fundo preto
    if n == 129 or n == 135 or n == 149 or n == 150 or n == 151 or n == 161 or n == 182:
        return 129
	
    # 153 - Triângulos virados para a esquerda com fundo preto 
	# 165 - Triângulos de Sierpinski com fundo preto
	# 183 - Triângulos com fundo branco
	# 195 - Triângulos virados para a direita com fundo preto

    ### class IV
    # 41
    # 54
    # 97
    # 106
    # 107
    # 110
    # 120
    # 121
    # 124
    # 137
    # 147
    # 169
    # 193
    # 225

    return n

###############################################################################
def neighborhood_number(line, width, i, r, k):
    # Toda vizinhança é o dobro do raio mais a célula central
    neighborhoodSize = int(2 * r + 1)

    if neighborhoodSize <= 1:
        return int(line[i])
    
    # A quantidade de células à esquerda. Se o tamanho for impar a quantidade
    # de células em torno da célula central são iguais, caso contrário a 
    # quantidada à esquarda é uma menor.
    if neighborhoodSize % 2 != 0:
        loffset = int(neighborhoodSize / 2)
    else:
        loffset = int(neighborhoodSize / 2 - 1)

    roffset = int(neighborhoodSize / 2)

    # 
    if neighborhoodSize < width:
        ini = int(i - loffset)
        fim = int(i + roffset + 1)

        if ini < 0: 
            ini = width + ini
        elif ini >= width:
            ini = ini - width

        if fim > width:
            fim = fim - width
        elif fim < 0:
            fim = width - fim
    #
    else:
        ini = 0
        fim = width
        neighborhoodSize = width

    result = 0
    potencia = neighborhoodSize - 1

    while potencia >= 0:
        result += line[ini] * int(pow(k, potencia))

        ini += 1
        potencia -= 1

        if ini < 0:
            ini = width + ini
        elif ini >= width:
            ini = ini - width

    return result

###############################################################################
def get_plain_cellularautomata(n, k, r, t, transient=0, ic=None):
    length = t - transient
    
    if ic == None:
        ic = [rnd.randrange(0,k,1) for _ in range(length)]

    return ca.plain(
        ca.cellularautomata(n, k, r, ic, t, transient)
    )
    
        
###############################################################################
def retrieve_block_window(te, size, X, Y, J, I):
    windows = []
 
    s = 0
    while s < size:
        window = ca.split([0]*(I*J), J)

        x = rd.randint(0, X)
        y = rd.randint(0, Y)

        if X - x < J:
            x = X - J

        if Y - y < I:
            y = Y - I

        i = 0
        while i < I:
            j = 0
            while j < J:
                window[i][j] = te[y + i][x + j]
            
                j += 1
            
            i += 1
        
        windows.append(window)

        s += 1
    
    return windows

###############################################################################
def retrieve_windows_tes(k, r, length, t, transient, wini, winj, output_size, subclass=False):
    batch = [[],[]]

    X = length
    Y = t - transient
    J = winj
    I = wini

    if output_size == 256:
        rule = rnd.randrange(0, output_size, 1)
        n = rule

        if subclass:
            rule = get_subclass(rule)
    elif output_size == 4:
        rule = int(rnd.randrange(0, 4, 1))
        classes = WolframClasses[rule]
        n = classes[rnd.randrange(0, len(classes), 1)]

    te = ca.cellularautomata(n, k, r, 
        [rnd.randrange(0,k,1) for _ in range(length)], 
        t, transient
    )

    blocks = retrieve_block_window(te, 50, X, Y, J, I)
    len_blocks = len(blocks)

    a = 0
    while a < len_blocks:
        y = [0] * output_size
        y[rule - 1] = 1
        
        x = ca.plain(blocks[a])
        
        batch[0].append(x)
        batch[1].append(y)

        a += 1

    return batch

###############################################################################
def retrieve_temporal_evolutions(k, r, length, t, transient, output_size, subclass=False):
    rule = 0
    a = 0
    batch = [[],[]]

    while a < 50:
        rule = rnd.randrange(0, output_size, 1)
        n = rule

        if subclass:
            rule = get_subclass(rule)
      
        y = [0] * output_size
        y[rule] = 1

        x = ca.plain(
            ca.cellularautomata(n, k, r, 
                [rnd.randrange(0,k,1) for _ in range(length)], 
                t, transient
            )
        )
      
        batch[0].append(x)
        batch[1].append(y)

        a += 1

    return batch

################################################################
def retrieve_entropy_evolution(n, k, r, width, t, transient=0, w=10, ic=None):
    if ic == None:
        ic = [rnd.randrange(0,k,1) for _ in range(width)]

    te = ca.cellularautomata(n, k, r, ic, t + w, transient)

    entropies = [0] * (len(te) - w)

    l = 0
    while l < (len(te) - w):
        entropies[l] = evolution_entropy(k, r, w, l, te)

        l += 1

    return entropies, n

###############################################################################
def get_neighborhood(i, r, line):
    m = 2*r + 1
    result = []

    ini = i - r
    fim = i + r

    if ini >= 0 and fim < len(line):
        return line[ini:fim + 1]

    if ini < 0:
        ini += len(line)
    
    if fim >= len(line):
        fim -= len(line)

    while ini != fim + 1:
        result.append(line[ini])

        ini += 1
        if ini >= len(line):
            ini -= len(line)
        
    return result

###############################################################################
def frequency_of_symbols(k, r, w, t, te):
    J = len(te[0])
    symbols = {}

    j = 0
    while j < J:
        line = []

        i = t
        while i < t + w:
            line = line + get_neighborhood(j, r, te[i])
            i += 1
            
        s = neighborhood_number(line, len(line), int(len(line) / 2), int(len(line) / 2), k)

        try:
            symbols[s] += 1
        except KeyError:
            symbols[s] = 1

        j += 1

    return list(symbols.values())

###############################################################################
def entropy(values):
    total = sum(values)
    H = -sum([(i/total) * np.log2(i/total) for i in values])

    return H

###############################################################################
def evolution_entropy(k, r, w, t, te):
    frequency_symbols = frequency_of_symbols(k, r, w, t, te)
    frequency_symbols = [i for i in frequency_symbols if i != 0]

    frequency_symbols = (frequency_symbols / np.sum(frequency_symbols))

    return np.sum(-(frequency_symbols * np.log2(frequency_symbols)))

###############################################################################
def word_entropy(k, r, w, te):
    frequency_word = {}

    for line in te:
        symbols = frequency_of_symbols(k, r, w, 0, [line])
        key = from_digits(b, symbols)

        try:
            frequency_word[key] += 1
        except KeyError:
            frequency_word[key] = 1

    return entropy(frequency_word.values())

###############################################################################
def tranform2rgb(evolution_entropy_list):
    max_rgb_value = int(pow(2, 24))
    
    rgb = []
    rgbvalue = 0

    for line in evolution_entropy_list:
        rgbline = []

        for proportion in line:
            rgbvalue = int(proportion * max_rgb_value)

            rgbline.append(rgbvalue & 255)
            rgbline.append((rgbvalue >> 8) & 255)
            rgbline.append((rgbvalue >> 16) & 255)
        
        rgb.append(rgbline)

    return rgb

###############################################################################
def evolution_entropy_rgb(n, k, r, width, T, transient, w, b, step_proportion):
    # Compute the max value for initial condition
    initialConditionMax = int(pow(k, width))
    # Define the step forward to initial condition value.
    # The proportion is compute under max value
    stepForwardPercent = int(step_proportion * initialConditionMax)

    # Set the first value for initial condition
    initialConditionVal = 0
    # Create list
    entropy_arraies = []

    while initialConditionVal < initialConditionMax:
        ic = from_number_fix(initialConditionVal, k, width)

        entropy_arraies.append(
                [evolution_entropy(k, r, w, 0, [line]) for line in ca.cellularautomata(n, k, r, ic, T, transient)]
            )

        initialConditionVal += stepForwardPercent

        if initialConditionVal > initialConditionMax:
            initialConditionVal = initialConditionMax

    rgbarray = tranform2rgb(entropy_arraies)
    
    return rgbarray

###############################################################################
def array_of_complex(fft):
  '''
  Put in arrays the result of Fast Fourier Transform.
  '''

  fft_x = []
  fft_y = []

  for line in fft:
    for c in line:
      fft_x.append(c.real)
      fft_y.append(c.imag)

  return fft_x, fft_y

###############################################################################
def fourier_spectrum(fft):
  fourierSpectrum = []

  for i in range(len(fft)):
    fourierSpectrum.append([0] * len(fft[i]))

    for j in range(len(fft[i])):
      fourierSpectrum[i][j] = np.sqrt(
        fft[i][j].real*fft[i][j].real + fft[i][j].imag*fft[i][j].imag)

  return fourierSpectrum
  
###############################################################################
def power_spectrum(fft):
  spectrum = []

  for i in range(len(fft)):
    spectrum.append([0] * len(fft[i]))

    for j in range(len(fft[i])):
      spectrum[i][j] = float(fft[i][j].real*fft[i][j].real + 
        fft[i][j].imag*fft[i][j].imag)

  return spectrum
  
###############################################################################
def get_fft(et):
  fft = np.fft.fft2(et)

  return fft

###############################################################################
def date_log():
    return str(tm.asctime( tm.localtime(tm.time()) ))

###############################################################################
def entropy_evolution_elementar_space():
    k = 2
    r = 1
    width = 100
    T = 150 #3*width
    transient = 50 #2*width

    w = 1
    ic = [rd.randrange(0,k) for _ in range(width)]

    mean_length = 300
    class_number = 0

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

#------------------------------------------------------------------------------
if __name__ == '__main__':
    #entropy_evolution_elementar_space()

    y_result = [0 for _ in range(4)]
    y_result[wolfram_class(110) - 1] = 1

    print(y_result)