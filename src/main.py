from Algorithms.RNN import RNN
from DatasetsConsumers.Newsgroups import Newsgroups
from DatasetsConsumers.SpamHam import SpamHam
from DatasetsConsumers.Spamassassin import Spamassassin
from DatasetsConsumers.Trustpilot import Trustpilot

from Glove.glovemodel import GloVe
from rootfile import ROOTPATH
from utility.confusmatrix import plot_confusion_matrix
from utility.plotter import plot_data

# Load dataset
# datasets = [SpamHam(), Newsgroups(), Spamassassin(), Trustpilot()]
# algorithms = [RNN, SVM]

Dataset_Consumer = SpamHam()
emails, labels = Dataset_Consumer.load(True)
# Load GloVe model
GloVe_Obj = GloVe(200)
features = GloVe_Obj.get_features(emails, Dataset_Consumer)
# input_dim = GloVe_Obj.dimensionCount  # input dimension

# concat_emails = [' '.join(email) for email in emails]

#vectors = TfidfVectorizer()
#tfidf = vectors.fit_transform(concat_emails)
#features = np.array([size_control(sentence.data, 512) for sentence in tfidf])

# input_dim = 1024
# tfidf = compute_tfidf(Dataset_Consumer.word_count_list, emails)
# features = multi_size_control(tfidf, input_dim)
n_categories = len(set(labels))

Parameters = {
    'batch_size': 1000,
    'num_epochs': 50,
    'hidden_dim': 128,
    'layer_dim': 1,
    'learning_rate': 0.01,
    'output_dim': n_categories,
    'input_dim': features.shape[1]
}
PlotData, y_test, preds = RNN.run_train(Dataset_Consumer, features, labels, Parameters)
for plotClass in PlotData:
    plot_data(plotClass, True)


labs = Dataset_Consumer.get_subdirectories(ROOTPATH + "/data/20Newsgroups/")
plot_confusion_matrix(y_test, preds, labs)
