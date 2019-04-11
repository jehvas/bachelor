from keras_preprocessing import sequence
from keras_preprocessing.text import Tokenizer
from sklearn.metrics import precision_recall_fscore_support

from Algorithms.MLPT import MLP_tensorflow
from DatasetsConsumers.Newsgroups import Newsgroups
from Glove.glovemodel import GloVe
from utility.Parameters import get_params
from utility.confusmatrix import plot_confusion_matrix
from utility.plotter import plot_data


def run_all():
    # Load dataset
    datasets = [Newsgroups()]
    algorithms = [MLP_tensorflow]

    for dataset_consumer in datasets:
        for algorithm in algorithms:
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
                                                                            parameters, None, None,
                                                                            emails)

            for plotClass in data_to_plot:
                plot_data(plotClass, True)

            precision, recall, fscore, support = precision_recall_fscore_support(y_test, rounded_predictions)
            print("\nPrecision: ", precision)
            print("\nRecall: ", recall)
            print("\nFscore: ", fscore)
            print("\n")
            print("Avg fScore:", (sum(fscore)/len(fscore)))

            plot_confusion_matrix(y_test, rounded_predictions, dataset_consumer, algorithm.get_name(), normalize=True)


run_all()