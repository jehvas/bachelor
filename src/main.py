import sys
from sklearn.metrics import precision_recall_fscore_support
from Algorithms.MLPT import MLP_tensorflow, RNN_tensorflow, Bi_LSTM_tensorflow
from Algorithms.Perceptron import Perceptron
from Algorithms.SVM import SVM
from DatasetsConsumers.EnronEvidence import EnronEvidence
from DatasetsConsumers.EnronFinancial import EnronFinancial
from DatasetsConsumers.Newsgroups import Newsgroups
from DatasetsConsumers.Spamassassin import Spamassassin
from DatasetsConsumers.Trustpilot import Trustpilot
from Glove.glovemodel import GloVe
from utility.Parameters import get_params
from utility.confusmatrix import plot_confusion_matrix
from utility.plotter import plot_data

algorithms = {
    "all": [SVM, Perceptron, MLP_tensorflow, RNN_tensorflow, Bi_LSTM_tensorflow],
    "svm": [SVM],
    "perceptron": [Perceptron],
    "mlp": [MLP_tensorflow],
    "rnn": [RNN_tensorflow],
    "bi-lstm": [Bi_LSTM_tensorflow]
}
newsgroup = Newsgroups()
datasets = {
    "all": [Newsgroups(), Spamassassin(), EnronEvidence(), EnronFinancial(), Trustpilot()],
    "newgroups": [Newsgroups()],
    "spamassassin": [Spamassassin()],
    "enronevidence": [EnronEvidence()],
    "enronfinancial": [EnronFinancial()],
    "trustpilot": [Trustpilot()]
}

datasets_to_use = [Spamassassin()]
algorithms_to_use = [MLP_tensorflow]
# Check arguments
if len(sys.argv) != 3:
    print("There must be 2 arguments: an algorithm and a dataset.")
    print("Possible algorithms:")
    for x in algorithms.keys():
        print("\t" + x)
    print("Possible datasets:")
    for x in datasets.keys():
        print("\t" + x)
    # exit()
else:
    algorithms_to_use = algorithms[sys.argv[1]]
    datasets_to_use = datasets[sys.argv[2]]

for dataset_consumer in datasets_to_use:
    for algorithm in algorithms_to_use:
        emails, labels = dataset_consumer.load(True)
        glove = GloVe(200)
        features = glove.get_features(emails, dataset_consumer)
        print("Running algorithm:", algorithm.get_name())
        parameters = get_params(algorithm.get_name(), dataset_consumer)

        parameters['output_dim'] = len(set(labels))
        parameters['input_dim'] = features.shape[1]

        """
        dataset_consumer.setVocabulary(emails)
        max_words = 50000
        max_len = parameters['max_len']
        tok = Tokenizer(num_words=max_words)
        tok.fit_on_texts(emails)
        sequences = tok.texts_to_sequences(emails)
        sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)
        
        matrix = glove.get_weights_matrix(tok)
        """

        data_to_plot, y_test, rounded_predictions = algorithm.run_train(dataset_consumer, features, labels,
                                                                        parameters)

        for plotClass in data_to_plot:
            plot_data(plotClass, True)

        precision, recall, fscore, support = precision_recall_fscore_support(y_test, rounded_predictions)
        print("\nPrecision: ", precision)
        print("\nRecall: ", recall)
        print("\nFscore: ", fscore)
        print("\n")
        print("Avg fScore:", (sum(fscore)/len(fscore)))

        plot_confusion_matrix(y_test, rounded_predictions, dataset_consumer, algorithm.get_name(), normalize=True)