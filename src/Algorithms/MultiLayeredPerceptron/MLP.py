import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split as tts
from torch import optim

from utility.plotter import PlotClass


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        # self.embedding = nn.Embedding(len(Dataset_Consumer.vocabulary), GloVe_Obj.dimensionCount)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x = self.embedding(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def get_name():
    return 'MLP'


def uniqiueapsodjaapskdj():
    pass


def run_train(dataset, features, labels, parameters):
    batch_size = parameters['batch_size']
    num_epochs = parameters['num_epochs']
    hidden_dim = parameters['hidden_dim']
    layer_dim = parameters['layer_dim']
    learning_rate = parameters['learning_rate']
    output_dim = parameters['output_dim']
    input_dim = parameters['input_dim']

    x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2, random_state=1, stratify=labels)

    zippedtrain = list(zip(x_train, torch.Tensor(y_train).long()))
    zippedtest = list(zip(x_test, torch.Tensor(y_test).long()))

    trainloader = torch.utils.data.DataLoader(zippedtrain, batch_size=batch_size,
                                              shuffle=False)
    testloader = torch.utils.data.DataLoader(zippedtest, batch_size=batch_size,
                                             shuffle=False)

    model = Net(input_dim, hidden_dim, layer_dim, output_dim)

    criterion = nn.CrossEntropyLoss(weight=parameters['class_weights'])
    # criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_list = []
    iteration_list = []
    avg_accuracy_list = []
    min_accuracy_list = []
    max_accuracy_list = []
    for epoch in range(num_epochs):
        start_time = time.time()
        print("Epoch: ", epoch + 1, "/", num_epochs)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            model.zero_grad()
            inputs, train_labels = data
            inputs = inputs.float()
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        all_predictions = []
        for i, data in enumerate(testloader, 0):
            model.zero_grad()
            inputs, labels = data
            outputs = model(inputs.float())
            _, predicted = torch.max(outputs.data, 1)

            all_predictions = np.concatenate([all_predictions, predicted.numpy()])

        precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, all_predictions)
        accuracy = sum(fbeta_score) / len(fbeta_score)  # 100 * correct / float(total)

        min_accuracy_list.append(min(fbeta_score))
        avg_accuracy_list.append(accuracy)
        max_accuracy_list.append(max(fbeta_score))
        loss_list.append(loss.data)
        iteration_list.append(epoch)
        # Print Loss
        print(
            'Epoch: {} \t {:.2f}s Loss: {:.5f}  Accuracy: {:.9f} %'.format(epoch, time.time() - start_time, loss.item(),
                                                                           accuracy))

    plot_emails = []
    plot_labels = []
    final_preds = []
    for i, data in enumerate(testloader, 0):
        model.zero_grad()
        inputs, labels = data
        outputs = model(inputs.float())
        _, predicted = torch.max(outputs.data, 1)

        final_preds = np.concatenate([final_preds, predicted.numpy()])
        plot_labels.append(inputs)
        plot_emails.append(labels)

    return ([
                PlotClass([(iteration_list, loss_list)], "Number of epochs", "Loss", parameters, dataset, "MLP"),
                PlotClass([(iteration_list, min_accuracy_list), (iteration_list, avg_accuracy_list),
                           (iteration_list, max_accuracy_list)], "Number of epochs", "Accuracy",
                          parameters, dataset,
                          "MLP", ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
            ], y_test, all_predictions)
