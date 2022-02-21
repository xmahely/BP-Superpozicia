import math
import string

from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import normalizer as n
import random
from random import randint
import torch
import jaro_distance as distance
import pandas as pd

import numpy as np


all_letters = string.ascii_lowercase
all_letters = n.wordToAsciiValueList(all_letters)
num_letters = len(all_letters)
max_size = 38


def generateRandomly():
    idx = randint(1, 6)
    if idx == 1:
        tensor1, tensor2 = generate2IdenticalTensors()
    elif idx == 2 or idx == 3:
        tensor1, tensor2 = generate2SimilarTensors()
    else:
        tensor1, tensor2 = generate2RandomTensors()
    return tensor1, tensor2


def generate2IdenticalTensors():
    tensor1 = generateTensor()
    return tensor1, tensor1


def generate2SimilarTensors():
    tensor1 = generateTensor(tensor_size_min=6, tensor_size_max=38)
    size1 = list(tensor1.size())[0]
    size2 = randint(-4, 4) + size1
    tensor2 = torch.empty(size2)

    if size2 == size1:
        tensor2 = tensor1
        tensor2[randint(0, size2 - 1)] = tensor1[randint(0, size1 - 1)]
        tensor2[randint(0, size2 - 1)] = tensor1[randint(0, size1 - 1)]
    elif size2 > size1:
        for i in range(size1):
            tensor2[i] = tensor1[i]
        tensor2[size1] = all_letters[randint(0, num_letters - 1)]
        if size2 > size1 + 1:
            tensor2[size1+1] = all_letters[randint(0, num_letters - 1)]
        if size2 > size1 + 2:
            tensor2[size1+2] = all_letters[randint(0, num_letters - 1)]
        if size2 > size1 + 3:
            tensor2[size1+3] = all_letters[randint(0, num_letters - 1)]
    elif size1 > size2:
        for i in range(size2):
            tensor2[i] = tensor1[i]
        if size1 > size2 + 1:
            tensor2[randint(0, size2 - 1)] = all_letters[randint(0, num_letters - 1)]
        if size1 > size2 + 2:
            tensor2[randint(0, size2 - 1)] = all_letters[randint(0, num_letters - 1)]
        if size1 > size2 + 3:
            tensor2[randint(0, size2 - 1)] = all_letters[randint(0, num_letters - 1)]

    return tensor1, tensor2


def generate2RandomTensors():
    tensor1 = generateTensor()
    tensor2 = generateTensor()

    return tensor1, tensor2


def generateTensor(tensor_size_min=None, tensor_size_max=None, random_shuffle=True):
    if tensor_size_min is None and tensor_size_max is None:
        size = randint(3, max_size)
    elif tensor_size_min is None and tensor_size_max is not None:
        size = randint(3, tensor_size_max)
    elif tensor_size_min is not None and tensor_size_max is None:
        size = randint(tensor_size_min, max_size)
    else:
        size = randint(tensor_size_min, tensor_size_max)

    if random_shuffle is True:
        random.shuffle(all_letters)

    result = []
    for _ in range(size):
        i = randint(0, num_letters - 1)
        result.append(all_letters[i])

    return torch.tensor(result, dtype=torch.float)


def createTrainData(size):
    df = pd.DataFrame(columns=['names', 'similarity'])
    X = []
    y = []
    max_vector_len = 44

    for i in range(0, size):
        t1, t2 = generateRandomly()
        d = distance.JaroDistance(t1, t2)

        t1_size = list(t1.size())[0]
        t2_size = list(t2.size())[0]

        if max_vector_len - t1_size > 0:
            out = nn.ConstantPad1d((0, max_vector_len - t1_size), 0)
            t1 = out(t1)
        if max_vector_len - t2_size > 0:
            out = nn.ConstantPad1d((0, max_vector_len - t2_size), 0)
            t2 = out(t2)

        # names_tensor = pad_sequence([t1, t2]).t()
        names_tensor = torch.cat((t1, t2), 0)
        df.loc[i, 'names'] = names_tensor
        similarity = torch.tensor(d.getDistance())
        df.loc[i, 'similarity'] = similarity

        X.append(names_tensor)
        y.append(similarity)

    X = torch.stack(X, 0)
    y = torch.stack(y, 0)
    return X, y.reshape((y.shape[0], 1))