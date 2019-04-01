import numpy as np
import time
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.autograd import Variable
import torch.utils.data

from sklearn.model_selection import train_test_split as tts

from rootfile import ROOTPATH
from utility.confusmatrix import plot_confusion_matrix
from utility.plotter import PlotClass


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True,
                          nonlinearity='tanh').cuda()

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim).cuda()

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).float().cuda()
        # One time step
        out, hn = self.rnn(x, h0)
        out1 = out[-1, :, :]
        out = self.fc(out1)
        return out


def run_train(dataset, features, labels, parameters):
    print("Running algorithm: Algorithms.RNN")

    batch_size = parameters['batch_size']
    num_epochs = parameters['num_epochs']
    hidden_dim = parameters['hidden_dim']
    layer_dim = parameters['layer_dim']
    learning_rate = parameters['learning_rate']
    output_dim = len(set(labels))
    input_dim = features.shape[1]

    x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2, stratify=labels)

    # create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable
    features_train = torch.from_numpy(x_train)
    targets_train = torch.from_numpy(y_train).type(torch.LongTensor)  # data type is long

    # create feature and targets tensor for test set.
    features_test = torch.from_numpy(x_test)
    targets_test = torch.from_numpy(y_test).type(torch.LongTensor)  # data type is long

    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(features_train, targets_train)
    test = torch.utils.data.TensorDataset(features_test, targets_test)

    # data loader
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim).cuda()
    # Cross Entropy Loss
    error = nn.CrossEntropyLoss()

    # SGD Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    loss_list = []
    iteration_list = []
    avg_accuracy_list = []
    min_accuracy_list = []
    max_accuracy_list = []
    for epoch in range(num_epochs):
        start_time = time.time()
        for i, (tr_email, tr_labels) in enumerate(train_loader):
            train = tr_email.view(-1, tr_email.size()[0], input_dim).float().cuda()
            tr_labels = tr_labels.cuda()

            # Clear gradients
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(train)

            # Calculate softmax and ross entropy loss
            loss = error(outputs, tr_labels)

            # Calculating gradients
            loss.backward()

            # Update parameters
            optimizer.step()

        # Calculate Accuracy
        all_predictions = []
        for t_email, t_labels in test_loader:
            train = t_email.view(-1, t_email.size()[0], input_dim).float().cuda()

            # Forward propagation
            outputs = model(train)

            # Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]
            all_predictions = np.concatenate([all_predictions, predicted.cpu().numpy()])

        precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, all_predictions)
        accuracy = sum(fbeta_score) / len(fbeta_score)  # 100 * correct / float(total)

        # store loss and iteration
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
    for t_email, t_labels in test_loader:
        train = t_email.view(-1, t_email.size()[0], input_dim).float().cuda()
        # Forward propagation
        outputs = model(train)

        # Get predictions from the maximum value
        predicted = torch.max(outputs.data, 1)[1]
        final_preds = np.concatenate([final_preds, predicted.cpu().numpy()])
        plot_labels.append(t_labels)
        plot_emails.append(t_email)

    labs = dataset.get_subdirectories(ROOTPATH + "/data/20Newsgroups/")
    plot_confusion_matrix(y_test, all_predictions, labs)

    return [
        PlotClass([(iteration_list, loss_list)], "Number of epochs", "Loss",
                  "h_dim x n - {} x {}".format(hidden_dim, layer_dim), dataset,
                  "RNN"),
        PlotClass([(iteration_list, min_accuracy_list), (iteration_list, avg_accuracy_list),
                   (iteration_list, max_accuracy_list)], "Number of epochs", "Accuracy",
                  "h_dim x n - {} x {}".format(hidden_dim, layer_dim), dataset,
                  "RNN", ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ]
