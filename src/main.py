from Algorithms.MLPT import MLP_tensorflow
from Algorithms.Perceptron import Perceptron
from Algorithms.RNN import RNN
from Algorithms.SVM import SVM
from Algorithms.MultiLayeredPerceptron import MLP
from DatasetsConsumers.Newsgroups import Newsgroups
from DatasetsConsumers.SpamHam import SpamHam
from DatasetsConsumers.Spamassassin import Spamassassin
from DatasetsConsumers.Trustpilot import Trustpilot
from Glove.glovemodel import GloVe
from utility.Parameters import get_params
from utility.confusmatrix import plot_confusion_matrix
from utility.plotter import plot_data


def run_all():
    # Load dataset
    datasets = [Trustpilot(), SpamHam(), Newsgroups(), Spamassassin()]
    algorithms = [MLP_tensorflow]

    for dataset_consumer in datasets:
        for algorithm in algorithms:
            emails, labels = dataset_consumer.load(False)
            glove = GloVe(200)
            features = glove.get_features(emails, dataset_consumer)
            print("Running algorithm:", algorithm.get_name())
            parameters = get_params(algorithm.get_name(), dataset_consumer)

            parameters['output_dim'] = len(set(labels))
            parameters['input_dim'] = features.shape[1]

            data_to_plot, y_test, predictions = algorithm.run_train(dataset_consumer, features, labels, parameters)
            for plotClass in data_to_plot:
                plot_data(plotClass, True)

            plot_confusion_matrix(y_test, predictions, dataset_consumer, algorithm.get_name())


run_all()
