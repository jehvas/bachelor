import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from DatasetsConsumers.Newsgroups import Newsgroups

torch.manual_seed(1)

Dataset_Consumer = Newsgroups()
emails, labels = Dataset_Consumer.load(True)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


"""
word_to_ix = {}
for sent, labels in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
label_to_ix = {"DET": 0, "NN": 1, "V": 2}
"""

word_to_ix = Dataset_Consumer.vocabulary

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 64
HIDDEN_DIM = 64


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=0)
        return tag_scores


model = BiLSTM(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), Dataset_Consumer.n_categories)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
print("pre")
with torch.no_grad():
    inputs = prepare_sequence(emails[0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)
print("a")

batch_size = 100

emails_as_idx = np.asarray([prepare_sequence(email, word_to_ix) for email in emails])
# Pytorch train and test sets
train = torch.utils.data.TensorDataset(emails_as_idx, labels)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)

for epoch in range(1):
    for i in range(len(emails)):
        label = labels[i]
        email = emails[i]
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        word_indexes = prepare_sequence(email, word_to_ix)
        targets = label

        # Step 3. Run our forward pass.
        tag_scores = model(word_indexes)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(emails[0], word_to_ix)
    tag_scores = model(inputs)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(tag_scores)
