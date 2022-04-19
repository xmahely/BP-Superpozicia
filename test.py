# import random
# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
#
#
# class TestDataset(torch.utils.data.Dataset):
#
#     def __init__(self, X, y):
#         self.X = X
#         self.y = y
#         # if not torch.is_tensor(X) and not torch.is_tensor(y):
#         #     # Apply scaling if necessary
#         #     if scale_data:
#         #         X = StandardScaler().fit_transform(X)
#         #     self.X = torch.from_numpy(X)
#         #     self.y = torch.from_numpy(y)
#
#     def __len__(self):
#         return len(self.X)
#
#     def __getitem__(self, i):
#         return self.X[i], self.y[i]
#
#
# class MLP(nn.Module):
#     '''
#       Multilayer Perceptron for regression.
#     '''
#
#     def __init__(self):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(7, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1)
#         )
#
#     def forward(self, x):
#         '''
#           Forward pass
#         '''
#         return self.layers(x)
#
# def createTestData(count):
#     inputs = []
#     outputs = []
#     for i in range(count):
#         name = random.random()
#         last_name = random.random()
#         titles = random.randint(0, 1)
#         city = random.random()
#         region = random.random()
#         psc = random.choice([0, 0.2, 0.25, 0.4, 0.6, 0.8, 1])
#         domicile = random.randint(0, 1)
#         inputs.append([name, last_name, titles, city, region, psc, domicile])
#         # output = (name + last_name + 0.3 * city + 0.3 * region + 0.3 * psc + 0.3 * domicile) / 3.2
#         output = (name + last_name + 0.1 * city + 0.1 * region + 0.1 * psc + 0.1 * domicile) / 2.4
#         outputs.append(output)
#
#     input_tensor = torch.tensor(inputs,
#                                 dtype=torch.float64)
#     output_tensor = torch.tensor(outputs,
#                                  dtype=torch.float64)
#     return input_tensor, output_tensor
#
#
# def train():
#     train_losses = []
#     torch.manual_seed(42)
#     dataset_size = 10000
#     X, y = createTestData(dataset_size)
#     dataset = TestDataset(X, y)
#     batch_size = 100
#     epoch_size = round((dataset_size / (dataset_size / batch_size)))
#     trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
#
#     mlp = MLP()
#
#     loss_function = nn.L1Loss()
#     optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
#
#     for epoch in range(0, epoch_size):
#
#         print(f'Starting epoch {epoch + 1}')
#
#         current_loss = 0.0
#
#         for i, data in enumerate(trainloader, 0):
#             inputs, targets = data
#             inputs, targets = inputs.float(), targets.float()
#             targets = targets.reshape((targets.shape[0], 1))
#
#             optimizer.zero_grad()
#
#             outputs = mlp(inputs)
#
#             loss = loss_function(outputs, targets)
#
#             loss.backward()
#
#             optimizer.step()
#
#             current_loss += loss.item()
#             if (i + 1) % (dataset_size / batch_size) == 0:
#                 print('Loss after mini-batch: %.10f' %
#                       (current_loss / (dataset_size / batch_size)))
#                 train_losses.append(current_loss / (dataset_size / batch_size))
#                 current_loss = 0.0
#
#     print('Training process has finished.')
#
#     plt.plot(train_losses, '-o')
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.legend(['Train'])
#     plt.title('Train losses')
#
#     plt.show()
#
#     torch.save(mlp.state_dict(), 'model.pt')
#
#
# def main():
#     # train()
#     model_dict = torch.load('model.pt')
#     model = MLP()
#     model.load_state_dict(model_dict)
#
#
#     inpt1 = []
#     inpt1.append([1, 1, 1, 0.5, 0.5, 0.5, 1])
#     tens = torch.tensor(inpt1, dtype=torch.float64)
#     tens = tens.float()
#     test1 = model(tens)
#     print(test1)
#     inpt1.append([0.5, 0.3, 0.4, 0.5, 0.5, 0.5, 1])
#     tens = torch.tensor(inpt1, dtype=torch.float64)
#     tens = tens.float()
#     test2 = model(tens)
#     print(test2)
#
#
# if __name__ == "__main__":
#     main()
