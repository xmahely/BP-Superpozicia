from __future__ import unicode_literals, print_function, division
import numpy as np
from numpy import argmax
import torch
import torch.nn as nn
import torch.nn.functional as F
from io import open
import glob
import os
import MLP as mlp
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from pyjarowinkler import distance

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

MAX_LEN = 62


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def letterToIndex(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    if line is not None:
        tensor = torch.zeros(MAX_LEN, 1, n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][letterToIndex(letter)] = 1
        return tensor
    return None


def normalize(s):
    if s is not None:
        result = unicodeToAscii(s)
        return result.lower()
    return None


def temp(database, table):
    i = 0
    for row in table:
        if i > 10:
            break
        i = i + 1

        meno1 = lineToTensor(normalize(row[0].Meno))
        meno2 = lineToTensor(normalize(row[1].Meno))
        par_len = len(meno1) if len(meno1) > len(meno2) else len(meno2)
        a = lineToTensor(meno1)
        b = lineToTensor(meno2)

        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        similarity = cos(a[0:par_len], b[0:par_len]).mean()


        # database.insert_row_inp(row[0].CID, row[1].CID,
        #                         lineToTensor(normalize(row[0].Meno)), lineToTensor(normalize(row[1].Meno)),
        #                         lineToTensor(normalize(row[0].Priezvisko)), lineToTensor(normalize(row[1].Priezvisko)),
        #                         row[0].Pohlavie, row[1].Pohlavie,
        #                         lineToTensor(normalize(row[0].Tituly)), lineToTensor(normalize(row[1].Tituly)),
        #                         lineToTensor(normalize(row[0].Mesto)), lineToTensor(normalize(row[1].Mesto)),
        #                         lineToTensor(normalize(row[0].Kraj)), lineToTensor(normalize(row[1].Kraj)),
        #                         lineToTensor(row[0].PSC), lineToTensor(row[1].PSC),
        #                         lineToTensor(normalize(row[0].Danovy_Domicil)),
        #                         lineToTensor(normalize(row[1].Danovy_Domicil)))



# a = lineToTensor('95691')
# b = lineToTensor('95692')
#
# b = lineToTensor('Jone')
# # c = lineToTensor('Jodes')
# # d = lineToTensor('Johan')
# # e = lineToTensor('Martina')

if __name__ == '__main__':
    torch.set_printoptions(threshold=10_000)
    meno1 ='robert'
    meno2= 'roboberts'
    par_len = len(meno1) if len(meno1) > len(meno2) else len(meno2)
    a = lineToTensor(meno1)
    b = lineToTensor(meno2)
    # print(a)
    # print(a[0:par_len])
    # print(b[0:par_len])
    # print(a[1])
    # print(b)
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    print(cos(a[0:par_len], b[0:par_len]).mean())
    meno_dist = distance.get_jaro_distance(meno1, meno2, winkler=True, scaling=0.1)
    print(meno_dist)




# #
# #
# cos = nn.CosineSimilarity(dim=2, eps=1e-6)
# # dist = torch.cdist(a, e, p=2)
# #
# #
# #
# print(cos(a,b))
# # print(dist)

