#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""image_generator.py file.
    
Nome: Victor Augusto Alves Catanante
Número USP: 10839918
Código da Disciplina: SCC5830
Ano/Semestre: 2018/1
Título do Trabalho: Gerador de Imagens
"""

import numpy as np
import random

__version__ = '0.1'
__author__ = 'Victor Augusto'
__copyright__ = "Copyright (c) 2018 - Victor Augusto"

def main():
    """Método principal.
    
    Responsável por receber os parâmetros que serão utilizados na geração da imagem.
    """
    file_name = str(input()).rstrip()
    C = int(input())
    function_number = int(input())
    Q = int(input())
    N = int(input())
    B = int(input())
    S = int(input())

    raw_img = generate_raw_image(C, function_number, Q, N, S)

    digital_img = generate_digital_image(raw_img, C, N, B)

    rmse(file_name, digital_img, C, N, B)
    
def generate_raw_image(c_parameter, function_number, q_parameter, n_parameter, seed):
    """Método generate_digital_image.

    Método que realiza a geração de uma imagem inicial, em formato float, a partir dos parâmetros
    informados.
    
    :param c_parameter      O valor das dimensões da imagem da cena (C x C).
    :param function_number  O número da função matemática que será utilizada para gerar a imagem.
    :param q_parameter      O valor Q, arbitrário, para cálculos.
    :param n_parameter      O valor das dimensões da imagem digital (N x N).
    :param seed             O valor da semente S a ser utilizada nas funções de números aleatórios.
    :return                 A imagem em formato float resultante das funções.
    """

    raw_image = None

    if function_number == 1:
        raw_image = scene_1(c_parameter)
        
    if function_number == 2:
        raw_image = scene_2(c_parameter, q_parameter)
        
    if function_number == 3:
        raw_image = scene_3(c_parameter, q_parameter)
        
    if function_number == 4:
        raw_image = scene_4(c_parameter, seed)
        
    if function_number == 5:
        raw_image = scene_5(c_parameter, seed)
        
    return raw_image
    
def generate_digital_image(raw_image, C, N, B):
    """Método generate_digital_image.

    Método que realiza a geração de uma imagem digital, em formato int, a partir dos parâmetros
    informados.
    
    :param raw_image    A imagem em formato float resultante das funções.
    :param C            O valor das dimensões da imagem da cena (C x C).
    :param N            O valor das dimensões da imagem digital (N x N).
    :param B            O valor referente à quantidade de bits que será considerada.
    :return             A imagem digital, em formato uint8, para comparação.
    """
    
    # normalização da imagem para [0; 2^8 - 1]
    tmp_img = convert_to_uint8(raw_image)

    digital_image = np.zeros([N,N], np.uint8)

    d = int(C / N)
    
    for x in range(N):
        for y in range(N):
            inf_lim_x = np.multiply(x, d)
            inf_lim_y = np.multiply(y, d)
                
            # máximo local            
            digital_image[x,y] = np.max(tmp_img[inf_lim_x : np.add(inf_lim_x, d), inf_lim_y: np.add(inf_lim_y, d)])
    
    # selecionar os bits informados na entrada
    digital_image = digital_image >> (8 - B)
    
    return digital_image
    
def convert_to_uint8(img):
    """Método convert_to_uint8.
    
    Método que converte os valores float da imagem para
    valores inteiros na faixa de 0 a 255.
        
    :param img      A imagem original no formato float.
    :return         A imagem convertida.
    """
    
    converted_img = np.empty(img.shape, np.uint8)
    converted_img = img * 255.0/np.max(img) 
    return converted_img.astype(np.uint8)
    
def convert_to_uint16(img):
    """Método convert_to_uint16.
    
    Método que converte os valores float da imagem para
    valores inteiros na faixa de 0 a 65535.
        
    :param img      A imagem original no formato float.
    :return         A imagem convertida.
    """
    
    converted_img = np.empty(img.shape, np.uint16)
    converted_img = img * 65535.0/img.max()
    return converted_img.astype(np.uint16)
    
def scene_1(C):
    """Método scene_1.
    
    Método que gera uma imagem a partir da função:
    
    f(x, y) = (x + y)
    
    :param C        O valor das dimensões da imagem da cena (C x C).
    :return         A imagem em formato float.
    """
    
    img = np.empty([C, C], np.float32)
    
    n = C - 1
    
    for x in range(n):
        for y in range(n):
            img[x, y] = np.add(x, y)
    return img

def scene_2(C, Q):
    """Método scene_2.
    
    Método que gera uma imagem a partir da função:
    
    f(x, y) = |sin(x/Q) + sin(y/Q)|
    
    :param C        O valor das dimensões da imagem da cena (C x C).
    :param Q        O valor Q, arbitrário, para cálculos.
    :return         A imagem em formato float.
    """
    
    img = np.zeros([C, C], np.float32)
    
    for index in np.ndindex(C, C):
        img[index] = np.absolute(np.sin(index[0] / Q) + np.sin(index[1] / Q))

    return img      

def scene_3(C, Q):
    """Método scene_3.
    
    Método que gera uma imagem a partir da função:
    
    f(x, y) = (x/Q) − (y/Q)^1/2
    
    :param C        O valor das dimensões da imagem da cena (C x C).
    :param Q        O valor Q, arbitrário, para cálculos.
    :return         A imagem em formato float.
    """
    
    img = np.empty([C, C], np.float32)
    
    for index in np.ndindex(C, C):
        img[index] = np.absolute((index[0] / Q) - np.sqrt(index[1] / Q))

    return img     

def scene_4(C, S):
    """Método scene_4.
    
    Método que gera uma imagem a partir da função:
    
    f(x, y) = rand(0, 1, S)
    
    :param C        O valor das dimensões da imagem da cena (C x C).
    :param S        O valor da semente S a ser utilizada nas funções de números aleatórios.
    :return         A imagem em formato float.
    """
    
    img = np.empty([C, C], np.float32)
    random.seed(S)
    
    for x in range(C):
        for y in range(C):
            img[x, y] = random.random()
    return img     

def scene_5(C, S):
    """Método scene_5.
    
    Método que gera uma imagem a partir do algoritmo de passeio aleatório.
    
    :param C        O valor das dimensões da imagem da cena (C x C).
    :param S        O valor da semente S a ser utilizada nas funções de números aleatórios.
    :return         A imagem em formato float.
    """
    
    img = np.zeros([C, C], np.float32)
    
    # inicialização da semente S
    random.seed(S)
    
    # definição no número total de passos
    total_steps = 1 + int(np.square(C) / 2)

    img[0, 0] = 1.0
    
    x = y = dx = dy = 0
    
    for steps in range(1, total_steps):
        dx = random.randint(-1, 1)
        x = (x + dx) % C
        img[x, y] = 1.0
        
        dy = random.randint(-1, 1)
        y = (y + dy) % C
        img[x, y] = 1.0

    return img

def rmse(file_name, img, C, N, B):
    """Método rmse.

    Computa o valor da equação raiz do erro médio quadrático,
    de acordo com a imagem produzida e a imagem de entrada.
    
    :param file_name    A string com o nome do arquivo que possui a imagem de referência.
    :param C            O valor das dimensões da imagem da cena (C x C).
    :param N            O valor das dimensões da imagem digital (N x N).
    :param B            O valor referente à quantidade de bits que será considerada.
    """
    
    # inicialização da imagem de referência.
    R = np.load(file_name)
    
    # normalização da imagem para [0; 2^8 - 1]
    R = convert_to_uint8(R)
    
    # selecionar os bits informados na entrada
    R = R >> (8 - B)
    
    squared_sum = .0
    for index in np.ndindex(N, N):
        squared_sum += np.square(float(img[index]) - float(R[index]))

    ans = np.sqrt(squared_sum)

    print ('{:.4f}'.format(ans))
    
if __name__ == "__main__":
    main()
