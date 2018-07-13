#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""super_res.py file.
    
Nome: Victor Augusto Alves Catanante
Número USP: 10839918
Código da Disciplina: SCC5830
Ano/Semestre: 2018/1
Título do Trabalho: Realce e Superresolução
"""

import numpy as np
import imageio

__version__ = '0.1'
__author__ = 'Victor Augusto'
__copyright__ = "Copyright (c) 2018 - Victor Augusto"

PNG = ".png"

def main():
    """Método principal.
    
    Responsável por receber os parâmetros que serão utilizados durante o processamento.

    imglow: nome do arquivo base para as imagens de baixa resolução.
    imghigh: nome do arquivo com imagem de alta resolução.
    option: método de realce, com opções: 0, 1, 2 e 3.
    gamma: parâmetro do método de realce 3 (default = 1).
    """
    imglow = str(input()).rstrip()
    imghigh = str(input()).rstrip()
    option = int(input())
    gamma = float(input())

    # Leitura da imagem de alta resolução.
    imghigh = (imageio.imread(imghigh + PNG))

    # Etapa de realce.
    enhanced_img_list = enhancement(imglow, option, gamma)

    # Etapa de superresolução.
    highest_res_img = superresolution(enhanced_img_list)

    # Comparação com a imagem de prova.
    rmse(imghigh, highest_res_img)

def enhancement(imglow, option, gamma=1.0):
    """Método enhancement.

    Método que realiza o realce (ou não), e posteriormente invoca
    o método de superresolução de acordo com os parâmetros informados.
    
    :param imglow       O nome do arquivo base para as imagens de baixa resolução.
    :param imghigh      O nome do arquivo com imagem de alta resolução.
    :param option       O método de realce, com opções: 0, 1, 2 e 3.
    :param gamma        O parâmetro do método de realce 3 (default = 1).
    :return             A imagem em formato float.
    """

    img_list = []

    # Leitura das imagens a partir dos arquivos.
    for i in range(1, 5):
        img_list.append(imageio.imread(imglow + str(i) + PNG))

    if option == 0:
        return img_list

    if option == 1:
        # Cálculo do histograma cumulativo para cada imagem.
        ha_list, bins = individual_cumulative_histogram(img_list)

        # Realização da equalização do histograma e interpolação dos valores na imagem.
        return multiple_eq_histogram_to_image(ha_list, bins, img_list)

    elif option == 2:
        # Cálculo do histograma cumulativo com todas as imagens.
        ha, bins = multiple_cumulative_histogram(img_list)

        # Realização da equalização do histograma e interpolação dos valores na imagem.
        return individual_eq_histogram_to_image(ha, bins, img_list)

    elif option == 3:
        # Realização do ajuste gamma em cada imagem.
        gamma_corrected_img_list = []
        for img in img_list:
            gamma_corrected_img_list.append(gamma_adjustment(img, gamma))

        return gamma_corrected_img_list

def superresolution(enhanced_img_list):
    """Método superresolution.

    Método que realiza o processo de superresolução das imagens.
    Composição da imagem a partir das imagens que sofreram realce.
    
    Os laços aninhados inserem nas linhas pares os elementos
            a[i, j]; c[i, j]
            
    e nas linhas ímpares os elementos
            b[i, j]; d[i, j]
            
    Para tal, são utilizadas variáveis auxiliares que crescem à metade
    do que crescem os índices x e y.

    :param enhanced_img_list    A lista com as imagens que sofreram
                                realce.
    :return                     A imagem após o processo de superreso-
                                lução.
    """
    # Extração das imagens da lista.    
    a = enhanced_img_list[0]
    b = enhanced_img_list[1]
    c = enhanced_img_list[2]
    d = enhanced_img_list[3]

    # Tupla com o tamanho da imagem de superresolução (dobro da original).
    size = tuple(np.multiply(a.shape, 2))
    
    img = np.zeros(size)
    
    N = a.shape[0]
    M = a.shape[1]
    
    aux_x = 0
    for x in range(N * 2):
        aux_y = 0
        if x != 0 and x % 2 == 0:
            aux_x += 1

        for y in range(M * 2):
            if x % 2 == 0:
                if y % 2 == 0:
                    if y != 0 and y % 2 == 0:
                        aux_y += 1
                    img[x, y] = a[aux_x, aux_y]
                else:
                    img[x, y] = c[aux_x, aux_y]
            else:
                if y % 2 == 0:
                    if y != 0 and y % 2 == 0:
                        aux_y += 1
                    img[x, y] = b[aux_x, aux_y]
                else:
                    img[x, y] = d[aux_x, aux_y]
    return img

def multiple_eq_histogram_to_image(ha_list, bins, img_list):
    """Método multiple_eq_histogram_to_image.

    Realiza o mapeamento dos valores dos histogramas para imagens
    base por meio da interpolação linear.

    :param ha_list      A lista que contém os quatro histogramas.
    :param bins         As faixas dos histogramas.
    :param img_list     A lista de imagens originais.
    :return             A lista de imagens equalizadas e interpoladas.
    """
    for i in range(4):
        img = img_list[i]
        ha = ha_list[i]
        
        # Interpolação linear.
        equalized_img = np.interp(img.flatten(), bins[:-1], ha)

        img_list[i] = equalized_img.reshape(img.shape)
    return img_list

def individual_eq_histogram_to_image(ha, bins, img_list):
    """Método individual_eq_histogram_to_image.

    Realiza o mapeamento dos valores de um histograma para uma
    imagem base por meio da interpolação linear.

    :param ha           O histograma.
    :param bins         As faixa do histograma.
    :param img_list     A lista de imagens originais.
    :return             A lista de imagens equalizadas e interpoladas.
    """
    for i in range(4):
        img = img_list[i]
        
        # Interpolação linear.
        equalized_img = np.interp(img.flatten(), bins[:-1], ha)
    
        img_list[i] = equalized_img.reshape(img.shape)
    return img_list

def individual_cumulative_histogram(img_list):
    """Método individual_cumulative_histogram.
    
    Realiza o cálculo do histograma cumulativo para cada uma das
    imagens da lista.

    :param img_list     A lista de imagens originais.
    :return             A lista de histogramas cumulativos e as faixas
                        utilizadas para cada histograma.
    """
    # Histogramas separados para cada imagem.
    ha_list = []
    for img in img_list:
        # Definição do histograma.
        h, bins = np.histogram(img.flatten(), bins=256, range=(0.0, 255.0), density=True)

        # Cálculo do histograma cumulativo.
        ha = h.cumsum() * 100

        # Normalização.
        ha = ha * 255.0 / ha.max()
        ha_list.append(ha)
        
    return ha_list, bins

def multiple_cumulative_histogram(img_list):
    """Método multiple_cumulative_histogram.
    
    Realiza o cálculo da média dos histogramas cumulativos
    para as imagens da lista.

    :param img_list     A lista de imagens originais.
    :return             O histograma cumulativo que representa
                        a média dos histogramas parciais calculados
                        e as faixas utilizadas para cada histograma.
    """
    # Histogramas separados para cada imagem.
    bins = []
    mean = np.empty([0])

    for img in img_list:
        # Definição do histograma.
        h, bins = np.histogram(img.flatten(), bins=256, density=True)
        
        # Cálculo do histograma cumulativo.
        ha = h.cumsum() * 100

        if mean.size == 0:
            mean = ha.astype('uint8')
        else:
            mean = np.abs(np.mean([mean, ha], axis=0))

    return mean, bins

def gamma_adjustment(img, gamma):
    """Método gamma_adjustment.
    
    Realiza o ajuste gamma sobre uma imagem.

    :param img      A imagem original.
    :param gamma    O parâmetro da função de ajuste gamma.
    :return         A imagem ajustada. 
    """
    gamma_corrected_img = np.zeros(img.shape)

    N = img.shape[0]
    for index in np.ndindex(N):
        gamma_corrected_img[index] = np.floor(255 * np.power((img[index] / 255.0), (1 / gamma)))
        
    return gamma_corrected_img

def rmse(imghigh, img):
    """Método rmse.

    Computa o valor da equação raiz do erro médio quadrático,
    de acordo com a imagem produzida e a imagem de alta resolução.
    
    :param imghigh    A imagem de alta resolução original.
    :param img        A imagem produzida.
    """
    
    # Dimensões da imagem de alta resolução.
    M = imghigh.shape[0]
    N = imghigh.shape[1]
    
    squared_sum = .0
    for index in np.ndindex(N, N):
        squared_sum += np.multiply(np.square(float(imghigh[index]) - float(img[index])), 1 / (M * N) )

    ans = np.sqrt(squared_sum)

    print ('{:.4f}'.format(ans))
    
if __name__ == "__main__":
    main()
