from collections import Counter

import numpy as np
import time
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.autograd import Variable
import torch.utils.data

from DatasetsConsumers.Newsgroups import Newsgroups
from DatasetsConsumers.SpamHam import SpamHam
from Glove.glovemodel import GloVe
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts

from utility.TFIDF import compute_tfidf


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True,
                          nonlinearity='relu')

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).float()
        # One time step
        out, hn = self.rnn(x, h0)
        out1 = out[-1, :, :]
        out = self.fc(out1)
        return out


print("Running algorithm: Algorithms.RNN")

# Load dataset
Dataset_Consumer = Newsgroups()
emails, labels = Dataset_Consumer.load(True)

# Load GloVe model
GloVe_Obj = GloVe(50)
features = GloVe_Obj.get_features(emails, Dataset_Consumer)

tfidf = compute_tfidf(Dataset_Consumer.word_count_list, emails)
print("PROGRAM WILL NOW EXIT (REMOVE THESE LINES)")
exit(0)
# Create training data
x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2, stratify=labels)

# batch_size, epoch and iteration
batch_size = 1000
num_epochs = 1

# create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable
featuresTrain = torch.from_numpy(x_train)
targetsTrain = torch.from_numpy(y_train).type(torch.LongTensor)  # data type is long

# create feature and targets tensor for test set.
featuresTest = torch.from_numpy(x_test)
targetsTest = torch.from_numpy(y_test).type(torch.LongTensor)  # data type is long

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest, targetsTest)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

# Create RNN
input_dim = GloVe_Obj.dimensionCount  # input dimension
hidden_dim = 100  # hidden layer dimension
layer_dim = 2  # number of hidden layers
output_dim = 20  # output dimension

model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)

# Cross Entropy Loss
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.05
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

seq_dim = 100
loss_list = []
iteration_list = []
avg_accuracy_list = []
min_accuracy_list = []
max_accuracy_list = []
count = 0
for epoch in range(num_epochs):
    startTime = time.time()
    for i, (email, labels) in enumerate(train_loader):
        train = Variable(email.view(-1, email.size()[0], input_dim)).float()
        labels = Variable(labels)

        # Clear gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(train)

        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)

        # Calculating gradients
        loss.backward()

        # Update parameters
        optimizer.step()

    print("Epoch finished in:", time.time() - startTime)
    # Calculate Accuracy
    # correct = 0
    # total = 0
    # Iterate through test dataset
    all_predictions = []
    for t_email, t_labels in test_loader:
        train = Variable(t_email.view(-1, t_email.size()[0], input_dim)).float()

        # Forward propagation
        outputs = model(train)

        # Get predictions from the maximum value
        predicted = torch.max(outputs.data, 1)[1]
        all_predictions = np.concatenate([all_predictions, predicted.numpy()])
        # Total number of labels
        # total += t_labels.size(0)

        # correct += (predicted == t_labels).sum()

    precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, all_predictions)
    accuracy = sum(fbeta_score) / len(fbeta_score)  # 100 * correct / float(total)

    # store loss and iteration
    min_accuracy_list.append(min(fbeta_score))
    avg_accuracy_list.append(accuracy)
    max_accuracy_list.append(max(fbeta_score))
    loss_list.append(loss.data)
    iteration_list.append(epoch)
    # Print Loss
    print('Epoch: {}  Loss: {}  Accuracy: {} %'.format(epoch, loss.item(), accuracy))

# visualization loss
plt.plot(iteration_list, loss_list)
plt.xlabel("Number of epochs")
plt.ylabel("Loss")
plt.title("RNN: Loss vs Number of epochs. Hidden Layer Size: {}x{}".format(hidden_dim, layer_dim))
plt.grid(True)
plt.show()

# visualization accuracy
plt.plot(iteration_list, min_accuracy_list, color="red")
plt.plot(iteration_list, avg_accuracy_list, color="blue")
plt.plot(iteration_list, max_accuracy_list, color="green")
plt.xlabel("Number of epochs")
plt.ylabel("Accuracy")
plt.title("RNN: Accuracy vs Number of epochs. Hidden Layer Size: {}x{}".format(hidden_dim, layer_dim))
plt.savefig('graph.png')
plt.grid(True)
plt.show()
