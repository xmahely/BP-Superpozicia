import os.path
import random

import numpy
import torch
from torch import nn
from torch.utils.data import DataLoader
import dataset as testdata
# import matplotlib.pyplot as plt
import MLP

from Levenshtein import distance as levenshtein
from Levenshtein import jaro_winkler

def createRandomData(count):
    inputs = []
    outputs = []
    for i in range(count):
        name = random.random()
        last_name = random.random()
        titles = random.randint(0, 1)
        dob = random.random()
        city = random.random()
        region = random.random()
        psc = random.choice([0, 0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 1])
        domicile = random.randint(0, 1)
        inputs.append([name, last_name, titles, dob, city, region, psc, domicile])

        # output = (name + last_name + 0.9*dob + 0.1 * city + 0.1 * region + 0.1 * psc + 0.1 * domicile) / 3.3
        output = 0.3 * name + 0.3 * last_name + 0.278 * dob + 0.0305 * city + 0.0305 * region + 0.0305 * psc + 0.0305 * domicile
        outputs.append(output)

    input_tensor = torch.tensor(inputs,
                                dtype=torch.float64)
    output_tensor = torch.tensor(outputs,
                                 dtype=torch.float64)
    return input_tensor, output_tensor

def createTestData():
    input1_names = ['martina', 'martina', 'martina', 'martina']
    input2_names = ['martina', 'martina', 'mata', 'fero']
    input1_last_names = ['mahelyova', 'mahelyova', 'mahelyova', 'mahelyova']
    input2_last_names = ['mahelyova', 'mahelyova', 'mahelova', 'testovaci']
    input1_titles = ['', '', '', '']
    input2_titles = ['', '', '', 'ing']
    input1_dob = ['1999-05-12', '1999-05-12', '1999-05-12', '1999-05-12']
    input2_dob = ['1999-05-12', '1999-05-12', '1999-05-12', '1968-04-03']
    input1_city = ['bratislava', 'bratislava', 'bratislava', 'bratislava']
    input2_city = ['bratislava', 'galanta', 'bratislava', 'bratislava']
    input1_region = ['bratislavsky', 'bratislavsky', 'bratislavsky', 'bratislavsky']
    input2_region = ['bratislavsky', 'trnavsky', 'bratislavsky', 'bratislavsky']
    input1_psc = ['83104', '83104', '83104', '83104']
    input2_psc = ['83104', '92401', '83104', '83104']
    input1_domicile = ['sk', 'sk', 'sk', 'sk']
    input2_domicile = ['sk', 'sk', 'sk', 'sk']

    original1 = [input1_names, input1_last_names, input1_titles, input1_dob, input1_city, input1_region, input1_psc, input1_domicile]
    original2 = [input2_names, input2_last_names, input2_titles, input2_dob, input2_city, input2_region, input2_psc, input2_domicile]
    inputs = []
    outputs = []

    for i in range(len(input1_names)):
        titles = 1
        name = jaro_winkler(input1_names[i], input2_names[i])
        last_name = jaro_winkler(input1_last_names[i], input2_last_names[i])
        dob = 1 - (levenshtein(input1_dob[i], input2_dob[i]) / max(len(input1_dob[i]), len(input2_dob[i])))
        city = jaro_winkler(input1_city[i], input2_city[i])
        region = jaro_winkler(input1_region[i], input2_region[i])
        psc = 1 - (levenshtein(input1_psc[i], input2_psc[i]) / max(len(input1_psc[i]), len(input2_psc[i])))
        domicile = 1 - (levenshtein(input1_domicile[i], input2_domicile[i]) / max(len(input1_domicile[i]), len(input2_domicile[i])))
        inputs.append([name, last_name, titles, dob, city, region, psc, domicile])
        print(name)
        print(0.3*name)

        output = 0.3 * name + 0.3 * last_name + 0.278 * dob + 0.0305 * city + 0.0305 * region + 0.0305 * psc + 0.0305 * domicile
        outputs.append(output)

    input_tensor = torch.tensor(inputs,
                                dtype=torch.float64)
    output_tensor = torch.tensor(outputs,
                                 dtype=torch.float64)
    return original1, original2, input_tensor, output_tensor




def check_if_mode_exists():
    return os.path.exists("model.pt")


def test():
    model_dict = torch.load('model.pt')
    model = MLP.MLP()
    model.load_state_dict(model_dict)
    original1, original2, X, y = createTestData()
    tensor = X.clone().detach()
    tensor = tensor.float()

    test_data = model(tensor)

    for a, data in enumerate(test_data):
        print("Zaznam 1: ", end='')
        for o1 in original1:
            print(o1[a], end=' ')
        print("\nZaznam 2: ", end='')
        for o2 in original2:
            print(o2[a], end=' ')
        print("\nVýsledok z MLP: ", data.item())
        print("Očakávaný výsledok: ", y[a].item())


def train():
    if not check_if_mode_exists():
        train_losses = []
        val_losses = []
        torch.manual_seed(42)
        epoch_size = 100

        X, y = createRandomData(10000)
        dataset = testdata.TestDataset(X, y)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True, num_workers=1)

        X, y = createRandomData(2000)
        dataset_val = testdata.TestDataset(X, y)
        validate_loader = torch.utils.data.DataLoader(dataset_val, batch_size=100, shuffle=True, num_workers=1)

        mlp = MLP.MLP()

        loss_function = nn.L1Loss()
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

        best_accuracy = 0.0

        for epoch in range(0, epoch_size):

            # print(f'Epocha {epoch + 1}')

            current_loss = 0.0
            current_val_loss = 0.0
            current_val_accuracy = 0.0
            total = 0

            # Treningový cyklus
            mlp.train()
            for i, data in enumerate(train_loader, 0):
                inputs, targets = data
                inputs, targets = inputs.float(), targets.float()
                targets = targets.reshape((targets.shape[0], 1))

                optimizer.zero_grad()
                outputs = mlp(inputs)

                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()
                current_loss += loss.item()

            train_loss_value = current_loss / len(train_loader)
            train_losses.append(train_loss_value)

            # Validačný cyklus
            mlp.eval()
            with torch.no_grad():
                for data in validate_loader:
                    inputs, targets = data
                    inputs, targets = inputs.float(), targets.float()
                    targets = targets.reshape((targets.shape[0], 1))

                    outputs = mlp(inputs)

                    val_loss = loss_function(outputs, targets)
                    current_val_loss += val_loss.item()
                    total += outputs.size(0)
                    current_val_accuracy += (numpy.around(targets, decimals=2) == numpy.around(outputs, decimals=2)) \
                        .sum().item()

            val_loss_value = current_val_loss / len(validate_loader)
            val_losses.append(val_loss_value)
            accuracy = (100 * current_val_accuracy / total)

            if accuracy > best_accuracy:
                torch.save(mlp.state_dict(), 'model.pt')
                best_accuracy = accuracy

            print('Epoch', epoch + 1,
                  ', Training Loss is: %.4f' % train_loss_value,
                  ', Validation Loss is: %.4f' % val_loss_value,
                  ', Accuracy is %.2f %%' % accuracy)

        print('Trénovací proces bol ukončený.')

        # plt.plot(train_losses, '-b')
        # plt.plot(val_losses, '-r')
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        # plt.legend(['Train', 'Validation'])
        # plt.title('Stratové funkcie')
        # plt.show()

        # plt.plot(train_losses, '-b')
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        # plt.legend(['Train'])
        # plt.title('Trénovacia stratová funkcia')
        # plt.show()

        # torch.save(mlp.state_dict(), 'model.pt')
    print("Sieť už je natrénovaná")
