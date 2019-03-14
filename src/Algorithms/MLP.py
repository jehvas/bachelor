import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from sklearn.model_selection import train_test_split as tts
from torch import optim

from DatasetsConsumers.Newsgroups import Newsgroups
from Glove.glovemodel import GloVe

# Load dataset
Dataset_Consumer = Newsgroups()
emails, labels = Dataset_Consumer.load(True)

# Load GloVe model
GloVe_Obj = GloVe("glove.6B.50d.txt")
features = GloVe_Obj.get_features(emails)

# Create training data & SVM Stuff
x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2)

n_inputs = x_train

zippedtrain = list(zip(x_train, y_train))
zippedtest = list(zip(x_test, y_test))
batch_size = 200

trainloader = torch.utils.data.DataLoader(zippedtrain, batch_size=batch_size,
                                          shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(zippedtest, batch_size=batch_size,
                                         shuffle=False, num_workers=4)


def run(_hidden_size, _num_epochs):
    input_size = 50
    hidden_size = _hidden_size
    output_size = 20
    num_epochs = _num_epochs

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size)
            self.fc4 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = nn.functional.relu(self.fc1(x))
            x = nn.functional.relu(self.fc2(x))
            x = nn.functional.relu(self.fc3(x))
            x = nn.functional.relu(self.fc4(x))
            return x

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        # print("Epoch: ", epoch + 1, "/", num_epochs)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.float()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
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

    print('Finished Training')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = net(inputs.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("nodes: ", hidden_size, ': Accuracy: %d %%' % (
            100 * correct / total))


for i in range(1, 20):
    run(i * 10, 40)

'''
# MLP Stuff
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(500), random_state=1)

print("\nStarting fitting")
clf.fit(x_train, y_train)

print("Fitting done")
predictions = clf.predict(x_test)

precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, predictions)
print("\n--- Results ---")
print("Precision: ", precision)
print()
print("\n\nRecall: ", recall)
print("\n\nF_score: ", fbeta_score)

'''
