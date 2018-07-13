#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""restoration.py file.
    
Nome: Victor Augusto Alves Catanante
Número USP: 10839918
Código da Disciplina: SCC5830
Ano/Semestre: 2018/1
Título do Trabalho: Restauração
"""

import numpy as np
import imageio

__version__ = '0.1'
__author__ = 'Victor Augusto'
__copyright__ = "Copyright (c) 2018 - Victor Augusto"

np.warnings.filterwarnings('ignore')

def main():
    """Método principal.
    
    Responsável por receber os parâmetros que serão utilizados durante o processamento.

    icomp: nome da imagem original para comparação (Icomp).

    inoisy: nome da imagem ruidosa para filtragem (Inoisy).

    method: opção de escolha do método 
        1 – filtro adaptativo de redução de ruído local;
        2 – filtro adaptativo da mediana;
        3 – filtro da média contra-harmônica;

    N: tamanho do filtro (N): n X n.
    
    filter_param: parâmetros específicos para os filtros:
        – para o método 1: o valor da distribuição de ruído (σ)
        – para o método 2: o tamanho máximo do filtro (M): m X m
        – para o método 3: a ordem do filtro (Q)
    """
    icomp = str(input()).rstrip()
    inoisy = str(input()).rstrip()

    comp = imageio.imread(icomp)
    noisy = imageio.imread(inoisy)

    # Imagem de saída.
    img_out = np.zeros(noisy.shape, np.float32)

    method = int(input())
    N = int(input())

    # Valor do lado do quadrado que representa a imagem.
    X, Y = noisy.shape

    # Elemento central do filtro
    a = int(N / 2)

    if method == 1:
        # Filtro adaptativo de redução de ruído local.
        img_out = adaptive_local_noise_filter(N, noisy, img_out, X, Y, a)

    if method == 2:
        # Filtro adaptativo da mediana.
        img_out = adaptive_median_filter(N, noisy, img_out, X, Y, a)

    if method == 3:
        # Filtro da média contra-harmônica.
        img_out = contra_harmonic_mean_filter(N, noisy, img_out, X, Y, a)

    # Comparação com a imagem original.
    rmse(comp, img_out)


def adaptive_local_noise_filter(N, noisy, img, X, Y, a):
    """Método adaptive_local_noise_filter.
    
    Aplica o filtro adaptativo de redução de ruído local, de tamanho N x N,
    na imagem ruidosa.

    :param N        Tamanho do filtro.
    :param noisy    Imagem ruidosa.
    :param img      Imagem de saída.
    :param X        Dimensão original em x da imagem.
    :param Y        Dimensão original em y da imagem.
    :param a        Índice do elemento central do filtro.
    :return         Imagem com redução de ruído.
    """
    # Valor σ da distribuição de ruído.
    sigma = float(input())

    # Image wrap dos elementos iniciais.
    noisy = image_wrap(N, noisy)

    # Variância do ruído da imagem.
    filter_variancy = np.square(sigma)

    # Para cada pixel:
    for x in np.arange(a, X - a):
        for y in np.arange(a, Y - a):
            # Definição da região.
            g = noisy[x - a: x + a + 1, y - a: y + a + 1]

            # Média local.
            local_mean = np.mean(g)

            # Variância local.
            var = np.sum((g - local_mean)**2)

            img[x - a, y - a] = g[a, a] - ((filter_variancy / var) * (g[a, a] - local_mean))
    return img

def adaptive_median_filter(N, noisy, img, X, Y, a):
    """Método adaptive_median_filter.
    
    Aplica o filtro adaptativo da mediana, de tamanho N x N, na imagem
    ruidosa.

    :param N        Tamanho do filtro.
    :param noisy    Imagem ruidosa.
    :param img      Imagem de saída.
    :param X        Dimensão original em x da imagem.
    :param Y        Dimensão original em y da imagem.
    :param a        Índice do elemento central do filtro.
    :return         Imagem com redução de ruído.
    """
    # Tamanho máximo do filtro.
    M = int(input())

    # Image wrap dos elementos iniciais.
    noisy = image_wrap(N, noisy)

    # Para cada pixel:
    for x in range(a, X - a):
        for y in range(a, Y - a):
            g = noisy[x - a: x + a + 1, y - a: y + a + 1]

            z_max = np.max(g)
            z_min = np.min(g)
            z_med = np.median(g)

            img[x - a, y - a] = recursive_step_A(M, N, g, img, z_med, z_min, z_max, x - a, y - a, a) 
    return img

def recursive_step_A(M, n, g, img, z_med, z_min, z_max, x, y, a):
    """Método recursive_step_A.
    
    Aplica o filtro adaptativo da mediana, de tamanho N x N, na imagem
    ruidosa.

    :param M        Tamanho máximo do filtro.
    :param n        Tamanho atual do filtro.
    :param g        Região de filtragem.
    :param img      Imagem de saída.
    :param z_med    Valor da mediana local.
    :param z_min    Valor mínimo local.
    :param z_max    Valor máximo local.
    :param x        Índice x para a imagem de saída.
    :param y        Índice y para a imagem de saída.
    :param a        Índice do elemento central do filtro.
    :return         Valor da mediana local ou uma chamada recursiva com o tamanho do filtro incrementado.
    """
    a1 = z_med - z_min
    a2 = z_med - z_max

    if a1 > 0 and a2 < 0:
        step_B(img, g, z_med, z_min, z_max, x, y, a)
    else:
        n += 1
        if n <= M:
            return recursive_step_A(M, n, g, img, z_med, z_min, z_max, x, y, a)
        else:
            return z_med

def step_B(img, g, z_med, z_min, z_max, x, y, a):
    """Método step_B.
    
    Realiza a segunda etapa do filtro adaptativo da mediana.
    
    :param img      Imagem de saída.
    :param g        Região de filtragem.
    :param z_med    Valor da mediana local.
    :param z_min    Valor mínimo local.
    :param z_max    Valor máximo local.
    :param x        Índice x para a imagem de saída.
    :param y        Índice y para a imagem de saída.
    :param a        Índice do elemento central do filtro.
    :return         Valor da mediana local ou o valor original da imagem.
    """
    b1 = g[a, a] - z_min 
    b2 = z_med - z_max

    if b1 > 0 and b2 < 0:
        return img[x, y]
    else:
        return z_med

def contra_harmonic_mean_filter(N, noisy, img, X, Y, a):
    """Método contra_harmonic_mean_filter.
    
    Aplica o filtro da média contra-harmônica de tamanho N x N na imagem
    ruidosa.

    :param N        Tamanho do filtro.
    :param noisy    Imagem ruidosa.
    :param img      Imagem de saída.
    :param X        Dimensão original em x da imagem.
    :param Y        Dimensão original em y da imagem.
    :param a        Índice do elemento central do filtro.
    :return         Imagem com redução de ruído.
    """
    # Ordem do filtro.
    Q = float(input())

    # Zero padding dos elementos iniciais.
    noisy = zero_padding(N, noisy)

    # Para cada pixel:
    for x in range(a, X - a):
        for y in range(a, Y - a):
            g = noisy[x - a: x + a + 1, y - a: y + a + 1]
            img[x - a, y - a] = np.sum(np.power(g, Q + 1)) / np.sum(np.power(g, Q))
    return img

def image_wrap(N, img):
    """Método image_wrap.
    
    Realiza a transformação da imagem em uma matriz circular.

    :param N        Tamanho do filtro.
    :param img      Imagem a receber o wrap.
    :return         Imagem transformada em matriz circular.
    """
    wrap = np.pad(img, int(N / 2), 'wrap')
    if N % 2 == 0:
        wrap = wrap[:, : wrap.shape[1] - 1]
    return wrap

def zero_padding(N, img):
    """Método zero_padding.
    
    Completa as zonas fora do limite da imagem com zeros.

    :param N        Tamanho do filtro.
    :param img      Imagem a receber o padding.
    :return         Imagem com seus limites preenchidos com zero.
    """
    zero_padding = np.pad(img, int(N / 2), 'constant')
    if N % 2 == 0:
        zero_padding = zero_padding[:, : zero_padding.shape[1] - 1]
    return zero_padding

def rmse(comp, img_out):
    """Método rmse.

    Computa o valor da equação raiz do erro médio quadrático,
    de acordo com a imagem restaurada e a imagem original
    
    :param comp        A imagem de original.
    :param img_out     A imagem restaurada.
    """
    ans = np.linalg.norm(comp - img_out) * (1 / img_out.size)

    print ('{:.4f}'.format(ans))

if __name__ == "__main__":
    main()
