import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split as tts
from torch import optim

from DatasetsConsumers.Newsgroups import Newsgroups
from DatasetsConsumers.SpamHam import SpamHam
from Glove.glovemodel import GloVe

# Load dataset
if __name__ == '__main__':
    Dataset_Consumer = Newsgroups()
    emails, labels = Dataset_Consumer.load(True)

    # Load GloVe model
    GloVe_Obj = GloVe(50)
    features = GloVe_Obj.get_features(emails, Dataset_Consumer)

    # Create training data & SVM Stuff
    x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2, random_state=1)
    n_inputs = x_train

    zippedtrain = list(zip(x_train, y_train))
    zippedtest = list(zip(x_test, y_test))
    batch_size = 200

    trainloader = torch.utils.data.DataLoader(zippedtrain, batch_size=batch_size,
                                              shuffle=False, num_workers=4)
    testloader = torch.utils.data.DataLoader(zippedtest, batch_size=batch_size,
                                             shuffle=False, num_workers=4)


    def run(_hidden_size, _num_epochs):
        input_size = GloVe_Obj.dimensionCount
        hidden_size = _hidden_size
        output_size = 20
        num_epochs = _num_epochs

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                self.relu = nn.ReLU()
                # self.embedding = nn.Embedding(len(Dataset_Consumer.vocabulary), GloVe_Obj.dimensionCount)
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                # x = self.embedding(x)
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                x = self.relu(x)
                x = self.fc3(x)
                return x

        net = Net()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

        predictions_total = []

        for epoch in range(num_epochs):  # loop over the dataset multiple times
            print("Epoch: ", epoch + 1, "/", num_epochs)
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                net.zero_grad()
                # get the inputs
                inputs, labels = data
                inputs = inputs.float()
                optimizer.zero_grad()
                outputs = net(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                '''if i % 200 == 0:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 200))
                    running_loss = 0.0'''

        # print('Finished Training')

        for i, data in enumerate(testloader, 0):
            net.zero_grad()
            inputs, labels = data
            outputs = net(inputs.float())
            _, predicted = torch.max(outputs.data, 1)

            predictions_total = np.concatenate([predictions_total, predicted.numpy()])

        precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, predictions_total,
                                                                                  average="micro")
        print("\n--- Results ---")
        print("Precision: ", precision)
        print("\n\nRecall: ", recall)
        print("\n\nF_score: ", fbeta_score)


    for i in range(1, 2):
        run(100, 25)
