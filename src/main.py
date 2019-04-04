from Algorithms.RNN import RNN
from DatasetsConsumers.Newsgroups import Newsgroups
from DatasetsConsumers.Trustpilot import Trustpilot
from Glove import glovemodel

from Glove.glovemodel import GloVe
from utility.TFIDF import compute_tfidf
from utility.plotter import plot_data

# Load dataset
from utility.sizeController import multi_size_control

Dataset_Consumer = Trustpilot()
emails, labels = Dataset_Consumer.load(True)
# Load GloVe model
GloVe_Obj = GloVe(200)
features = GloVe_Obj.get_features(emails, Dataset_Consumer)
input_dim = GloVe_Obj.dimensionCount  # input dimension

input_dim = 1024
#tfidf = compute_tfidf(Dataset_Consumer.word_count_list, emails)
#features = multi_size_control(tfidf, input_dim)

Parameters = {
    'batch_size': 200,
    'num_epochs': 10,
    'hidden_dim': 100,
    'layer_dim': 1,
    'learning_rate': 0.01,
    'output_dim': len(set(labels)),
    'input_dim': features.shape[1]
}
PlotData = MLP.run_train(Dataset_Consumer, features, labels, Parameters)
for plotClass in PlotData:
    plot_data(plotClass, True)
