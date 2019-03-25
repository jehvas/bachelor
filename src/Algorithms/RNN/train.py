import numpy as np
import torch
from data import *
import random
import time
import math
import os
from torch import nn
from torch.autograd import Variable

from Algorithms.RNN.model import RNN
from DatasetsConsumers.Chromium import Chromium
from DatasetsConsumers.Newsgroups import Newsgroups
from DatasetsConsumers.SpamHam import SpamHam
from Glove.glovemodel import GloVe
from sklearn.model_selection import train_test_split as tts

if torch.cuda.is_available():
    device = torch.device('cpu')
    # torch.cuda.device(device)
else:
    device = torch.device('cpu')
print("Device:", device)

n_epochs = 100
print_every = 1000
plot_every = 10
learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn

print("Running algorithm: RNN")
# Load dataset
Dataset_Consumer = SpamHam()
emails, labels = Dataset_Consumer.load(True)

# Load GloVe model
GloVe_Obj = GloVe(200)
features = GloVe_Obj.get_weights_matrix(Dataset_Consumer.vocabulary).to(device)

# features = GloVe_Obj.get_features(emails)

# Create training data & Algorithms.SVM Stuff
x_train, x_test, y_train, y_test = tts(emails, labels, test_size=0.2, stratify=labels)


def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1)  # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return category_i.item()


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


'''
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor
'''

n_letters = GloVe_Obj.dimensionCount
n_categories = max(labels) + 1
n_hidden = 512
rnn = RNN(n_letters, n_hidden, 1, n_categories)
rnn.to(device)

optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()


def train(category_tensor, line_tensor):
    # hidden = rnn.initHidden()
    optimizer.zero_grad()

    # for i in range(line_tensor.size()[0]):
    output, hidden = rnn(line_tensor)  # , hidden)

    # LOS LOSSOS FUNCTIONOS
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()

    return output, loss.data


# Keep track of losses for plotting
current_loss = 0
all_losses = []


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()


def size_control(index_vector):
    size = 100
    if len(index_vector) < size:
        index_vector *= math.ceil(size / len(index_vector))
    return index_vector[:size]


print("Building training data")
# Build data for RNN
x_training_data = [None] * len(x_train)
y_training_data = [None] * len(y_train)
for i in range(len(x_train)):
    email_as_vectors = [Dataset_Consumer.vocabulary[word] for word in x_train[i]]
    email_as_vectors = [features[w_i] for w_i in size_control(email_as_vectors)]

    # Formatting the line tensor into size < word_count x 1 x embedding_dim > (1 is because of bulk loading)
    line_tensor = email_as_vectors
    line_tensor = torch.stack(line_tensor).to(device)
    # line_tensor = torch.sum(line_tensor, 0)
    line_tensor = torch.unsqueeze(line_tensor, 1).to(device)
    x_training_data[i] = line_tensor.to(device)

    category = y_train[i]
    category_tensor = Variable(torch.LongTensor([category])).to(device)
    y_training_data[i] = category_tensor.to(device)

print("Begin running")
for epoch in range(0, n_epochs):
    # category, line, category_tensor, line_tensor = getRandomChromiumPair()  # randomTrainingPair()
    # print(category, category_tensor.size(), line, line_tensor.size())
    total_attempts = 0
    num_correct = 0
    for i in range(len(x_training_data)):
        output, loss = train(y_training_data[i], x_training_data[i])
        current_loss += loss

        # Print epoch number, loss, name and guess
        correct = ""
        total_attempts += 1
        guess_i = categoryFromOutput(output)
        category = y_training_data[i].item()
        if guess_i == category:
            correct = '✓'
            num_correct += 1
        else:
            correct = '✗ (%s)' % category

        if i % print_every == 0:
            print('%d %d%% (%s) %.4f %.2f%% / %s %s' % (
                total_attempts, epoch / n_epochs * 100, timeSince(start), loss, num_correct / total_attempts * 100,
                guess_i, correct))

        # Add current loss avg to list of losses
        if total_attempts % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    print("epoch", epoch, "finished")
    print("%.2f%%" % (num_correct / total_attempts * 100))

torch.save(rnn, 'char-rnn-classification.pt')
