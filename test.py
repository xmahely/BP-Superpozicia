import math
import string
import normalizer as n
import random
from random import randint
import torch
import jaro_distance as distance
import pandas as pd


all_letters = string.ascii_lowercase
all_letters = n.wordToAsciiValueList(all_letters)
num_letters = len(all_letters)


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
    tensor1 = generateTensor(tensor_size_min=6)
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
        size = randint(3, 40)
    elif tensor_size_min is None and tensor_size_max is not None:
        size = randint(3, tensor_size_max)
    elif tensor_size_min is not None and tensor_size_max is None:
        size = randint(tensor_size_min, 40)
    else:
        size = randint(tensor_size_min, tensor_size_max)

    if random_shuffle is True:
        random.shuffle(all_letters)

    result = []
    for _ in range(size):
        i = randint(0, num_letters - 1)
        result.append(all_letters[i])

    return torch.tensor(result, dtype=torch.float)


df = pd.DataFrame(columns=['col1', 'col2', 'similarity'])

for i in range(1, 50):
    t1, t2 = generateRandomly()
    d = distance.JaroDistance(t1, t2)
    df.loc[i, 'col1'] = t1
    df.loc[i, 'col2'] = t2
    df.loc[i, 'similarity'] = d.getDistance()

for index, row in df.iterrows():
    print(row['col1'], "\n + ", row['col2'], " -> ", row['similarity'])
