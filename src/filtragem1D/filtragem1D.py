#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""filtragem1D.py file.
    
Nome: Victor Augusto Alves Catanante
Número USP: 10839918
Código da Disciplina: SCC5830
Ano/Semestre: 2018/1
Título do Trabalho: Filtragem 1D
"""

import numpy as np
import imageio

__version__ = '0.1'
__author__ = 'Victor Augusto'
__copyright__ = "Copyright (c) 2018 - Victor Augusto"

np.warnings.filterwarnings('ignore')
PI = np.pi

def main():
    """Método principal.
    
    Responsável por receber os parâmetros que serão utilizados durante o processamento.

    file_name: nome da imagem para filtragem.
    num_filter: opção de escolha do filtro (1 – arbitrário, 2 – função Gaussiana).
    n: tamanho do filtro.
    w: parâmetro(s) que define(m) os pesos do filtro:
        – para o método 1, uma sequência de n pesos, w A1 , w A2 , . . . , w An ;
        – para o método 2, um único valor que representa o desvio padrão da
          distribuição, σ;
    domain: domínio da filtragem (1 – espacial, 2 — frequência).
    """
    file_name = str(input()).rstrip()
    num_filter = int(input())
    n = int(input())
    
    # Leitura do filtro.
    w = None
    if num_filter == 1:
        w = arbitrary_filter()
    else:
        w = gaussian_filter(n)

    domain = int(input())

    # Leitura da imagem original.
    img_in = imageio.imread(file_name)

    # Armazenamento do formato original do array.
    shape = img_in.shape

    img_out = None
    if domain == 1:
        # Convolução no domínio espacial.
        img_out = convolution_1D(img_in.flatten(), w)
    elif domain == 2:
        # Cálculo da DFT para a imagem 1D.
        tmp = DFT1D(img_in.flatten(), False)

        # Cálculo da DFT para o filtro.
        w = DFT1D(w, True)

        # Convolução no domínio da frequência e DFT inversa.
        img_out = IDFT1D(frequency_domain_convolution1D(tmp, w))
    
    # Transformação da imagem 1D para 2D.
    img_out = img_out.reshape(shape)
    
    # Comparação com a imagem de prova.
    rmse(shape, img_in, img_out)

def arbitrary_filter():
    """Método arbitrary_filter.

    Método de criação do filtro arbitrário.
    
    :return     O filtro 1D arbitrário.
    """
    w_line = str(input()).rstrip()

    # Leitura de cada elemento da linha por meio da função map
    # e posterior transformação em lista.
    w = list(map(float, w_line.split()))
    return np.matrix(w)

def gaussian_filter(n):
    """Método gaussian_filter.

    Método de criação do filtro gaussiano.
    
    :param n    O tamanho do filtro.
    :return     O filtro 1D gaussiano.
    """
    sigma = float(input())

    center = int(n / 2)
    w = []
    s = 0
    for i in range (n):
        # Cálculo da equação.
        el = (1 / (np.sqrt(2 * PI) * sigma)) * np.exp(-1 * (i * i) / 2 * sigma * sigma)

        # Validação dos números negativos do filtro, antes do elemento central.
        if n < center:
            w.append(-el)
        else:
            w.append(el)

        # Somatório dos elementos para a normalização.
        s += el
        
    # Normalização dos valores do filtro gaussiano.
    w /= s
    return np.matrix(w)

def convolution_1D(f, w):
    """Método convolution_1D.

    Método que realiza a convolução 1D em imagens "achatadas".
    
    :param f          A imagem original.
    :param w          O filtro w.
    :return           A imagem convoluída.
    """
    # Elemento central do filtro
    a = int(w.size/2)

    # Image wrap dos elementos iniciais
    f = np.append(f, f[0 : a])

    N = f.shape[0]

    # Obtém o filtro invertido.
    w_flip = np.flip(np.flip(w, 0) , 1)
    
    g = np.zeros([N - a], np.float32)

    # Para cada pixel:
    for x in range(a, N - a):
        sub_f = f[x - a: x + a + 1]
            
        # Cálculo de g em x.
        g[x - a] = np.sum(np.multiply(sub_f, w_flip))
    return g

def DFT1D(A, matrix=False):
    """Método DFT1D.

    Método que realiza a Transformada Discreta de Fourier
    em uma dimensão.
    
    :param A            A imagem original, "achatada".
    :param matrix       Booleano que informa qual o índice do shape deve ser
                        utilizado para calcular o valor de n (número de freqûencias).
    :return             A imagem convoluída no domínio de Fourier.
    """
    n = 0
    if matrix:
        n = A.shape[1]
    else:
        n = A.shape[0]
        
    x = u_range = np.arange(n)
    F = map(lambda u: np.sum(np.multiply(A, np.exp((-1j * 2 * np.pi * u*x) / n ))), u_range)
    return np.fromiter(F, np.complex64)

def IDFT1D(F):
    """Método IDFT1D.

    Método que realiza a operação inversa à Transformada Discreta de Fourier
    em uma dimensão.
    
    :param F            A imagem no domínio de fourier.
    :return             A imagem no domínio espacial.
    """
    n = F.shape[0]
    u = x_range = np.arange(n)
    A = map(lambda x: np.real(np.sum(np.multiply(F, np.exp((1j * 2 * np.pi * u*x) / n )))), x_range)
    return np.fromiter(A, np.float32) / n
	
def frequency_domain_convolution1D(g, w):
    """Método frequency_domain_convolution1D.

    Método que realiza a convolução de um filtro 1D em uma imagem
    "achatada".
    
    :param g            A imagem no domínio de fourier.
    :param g            O filtro.
    :return             A imagem convoluída.
    """
    # Criação de um array com o filtro para facilitar a inserção
    # de zeros.
    w = np.array(w)
    
    w_padded = np.zeros(g.size - w.size)
    w_padded = np.insert(w_padded, 0, w)
    return np.multiply(g, w_padded)

def rmse(shape, img_in, img_out):
    """Método rmse.

    Computa o valor da equação raiz do erro médio quadrático,
    de acordo com a imagem filtrada e a imagem original.
    
    :param shape      O formato da imagem original.
    :param img_in     A imagem original.
    :param img_out    A imagem resultante do processamento.
    """
    M = shape[0]
    N = shape[1]

    img_out = img_out.astype('uint8')
    
    squared_sum = .0
    for index in np.ndindex(N, N):
        squared_sum += np.square((img_in[index]) - (img_out[index])) * (1 / (M * N))

    ans = np.sqrt(squared_sum)

    print ('{:.4f}'.format(ans))
    
if __name__ == "__main__":
    main()
