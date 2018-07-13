#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""filtragem2D.py file.
    
Nome: Victor Augusto Alves Catanante
Número USP: 10839918
Código da Disciplina: SCC5830
Ano/Semestre: 2018/1
Título do Trabalho: Filtragem 2D
"""

import numpy as np
import imageio
import math

__version__ = '0.1'
__author__ = 'Victor Augusto'
__copyright__ = "Copyright (c) 2018 - Victor Augusto"

np.warnings.filterwarnings('ignore')

PI = np.pi

sobel_f_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
sobel_f_y = np.array([[1, 2 , 1], [0, 0, 0], [-1, -2, -1]])

def main():
    """Método principal.
    
    Responsável por receber os parâmetros que serão utilizados durante o processamento.

    file_name: nome da imagem para filtragem.
    
    method: opção de escolha do método
        1 – arbitrário;
        2 – laplaciana da gaussiana;
        3 – operador sobel;
    
    w_params: parâmetro(s) que define(m) os pesos do filtro:
        – para o método 1:
            * altura e largura do filtro (h e w )
            * h linhas com w números reais cada, que são os pesos do filtro
        – para o método 2:
            * tamanho do filtro (n)
            * desvio padrão da distribuição (σ)
        – para o método 3: Não há parâmetros
    
    pos: 4 números reais que dão as posições para realização de cortes
        (Hlb, Hub, Wlb e Wub)
    
    data_set: nome do arquivo .npy com o dataset
    
    data_set_labels: nome do arquivo .npy com as labels do dataset
    """
    file_name = str(input()).rstrip()

    # Leitura da imagem e criação da imagem resultante nula.
    img = imageio.imread(file_name)
    
    img_out = np.zeros(img.shape, np.complex128)

    method = int(input())

    if method == 1:
        # Definição do filtro
        w = arbitrary_filter()

        # Produto ponto-a-ponto no domínio da frequência.
        img_out = frequency_domain_convolution(img, w)

    if method == 2:
        # Definição do filtro
        w = laplacian_of_gaussian_filter()

        # Produto ponto-a-ponto.
        img_out = frequency_domain_convolution(img, w)

    if method == 3:
        # Convolução do filtro no domínio espacial.
        tmp = sobel(img)

        # FFT.
        img_out = np.fft.fft2(tmp)
    
    pos = str(input()).rstrip()
    pos = list(map(float, pos.split()))

    # # Slicing da matriz.
    slice_out = slicing(img_out, pos)

    data_set = str(input()).rstrip()
    data_set_labels = str(input()).rstrip()

    KNN(slice_out, data_set, data_set_labels)

def slicing(img_out, pos):
    """Método slicing.

    :param img_out      A imagem resultante da filtragem.
    :param pos          Parâmetros de multiplicação dos limites.
    :return             A imagem cortada.
    """
    H, W = img_out.shape

    # 1/4 da matriz original.
    slice_1 = img_out[0 : int(H / 2), 0 : int(W / 2)]

    H, W = slice_1.shape

    # Cálculo dos limites direcionais.
    Hlb = int(pos[0] * H)
    Hub = int(pos[1] * H)
    Wlb = int(pos[2] * W)
    Wub = int(pos[3] * W)

    slice_2 = slice_1[Hlb : Hub, Wlb : Wub]
    return slice_2

def KNN(img_out, data_set, data_set_labels):
    """Método KNN.

    :return     
    """
    # Transformação da imagem em 1D.
    vec_out = img_out.flatten()

    # Definição do dataset e dos labels.
    dataset = np.load(data_set)
    labels = np.load(data_set_labels)

    dist = np.zeros(dataset.shape[0], np.complex128)

    # Cálculo das distâncias euclidianas.
    for i in range(dataset.shape[0]):
        dist[i] = np.linalg.norm(vec_out - dataset[i], axis=0)
    
    idx = np.where(dist == dist.min())[0][0]

    print (labels[idx])
    print (idx)

def arbitrary_filter():
    """Método arbitrary_filter.

    Método de criação do filtro arbitrário.
    
    :return     O filtro 2D arbitrário.
    """
    w = []
    w_line = str(input()).rstrip()

    # Leitura de cada elemento da linha por meio da função map
    # e posterior transformação em lista.
    w_size = tuple(map(int, w_line.split()))
    
    i = 0
    while(i < w_size[0]):    
        w_line = str(input()).rstrip()
        w.append(list(map(float, w_line.split())))
        i += 1

    w = np.matrix(w)
    return w

def laplacian_of_gaussian_filter():
    """Método laplacian_of_gaussian_filter.

    Método de criação do filtro laplaciano da gaussiana.
    
    :return     O filtro laplaciano da gaussiana.
    """
    n = int(input())
    sigma = float(input())

    M, N = LoG_index(n)
    
    w = np.zeros([n,n], np.float)

    LoG = lambda x, y, sigma: -(1 / (np.power(sigma, 4) * PI)) * (1 - ((x**2 + y**2) / (2 * (sigma**2)))) * np.exp(-((x**2 + y**2) / (2 * (sigma**2))))
    
    for i in range(n):
        for j in range(n):
            w[i, j] = np.real(LoG(M[i, j], N[i, j], sigma))

    # Normalização dos valores para soma = 0
    pos = np.where(w > 0)
    pos_sum = np.sum(w[pos])

    neg = np.where(w < 0)
    neg_sum = np.sum(w[neg])

    corr = pos_sum + neg_sum
    corr /= neg[0].size
    
    w[neg] -= corr

    return w

def LoG_index(n):
    """Método LoG_index.

    Método de criação dos índices para filtro laplaciano da gaussiana.
    
    :param n    O tamanho do filtro.
    :return     Os índices do filtro laplaciano da gaussiana.
    """
    M = np.empty([n, n], np.float32)
    N = np.empty([n, n], np.float32)

    x = np.linspace(-5, 5, n, endpoint=True)
    y = np.linspace(5, -5, n, endpoint=True)

    for i in range(n):
        for j in range(n):
            M[i, j] = x[i]
            N[i, j] = y[i]

    return M.T, N

def sobel(img):
    """Método sobel.

    Método que aplica o filtro sobel na imagem.

    :param img          A imagem original.
    :return             A imagem convoluída.
    """
    # Zero-padding da imagem para realizar a convolução do filtro 3x3.
    padded_img = np.pad(img, (1), 'constant')

    img_x = convolution_2D(padded_img, sobel_f_x, img.shape)
    img_y = convolution_2D(padded_img, sobel_f_y, img.shape)

    img_out = np.sqrt(np.square(img_x) + np.square(img_y))

    return img_out
    
def convolution_2D(f, w, original_shape):
    """Método convolution_2D.

    Método que realiza a convolução 2D em imagens.
    
    :param f          A imagem original.
    :param w          O filtro w.
    :return           A imagem convoluída.
    """
    # Elemento central do filtro
    a = int(w.shape[0] / 2)

    N = original_shape[0]

    # Obtém o filtro invertido.
    w_flip = np.flip(np.flip(w, 0) , 1)
    
    g = np.zeros([N, N], np.float32)

    # Para cada pixel:
    for x in range(a, N - a):
        for y in range(a, N - a):
            sub_f = f[x - a: x + a + 1, y - a: y + a + 1]

            # Cálculo de g em x e y.
            g[x - a, y - a] = np.sum(np.multiply(sub_f, w_flip))
    return g

def frequency_domain_convolution(img, w):
    """Método frequency_domain_convolution.

    Método que realiza a convolução de um filtro em uma imagem.
    
    :param img          A imagem no domínio de Fourier.
    :param w            O filtro no domínio de Fourier.
    :return             A imagem convoluída.
    """
    w_pad = np.zeros(img.shape, np.complex128)

    for index in np.ndindex(w.shape):
        w_pad[index] = w[index]

    # FFT.
    W = np.fft.fft2(w_pad)
    tmp = np.fft.fft2(img)

    return np.multiply(tmp, W)

if __name__ == "__main__":
    main()
