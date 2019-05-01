from collections import Counter

from sklearn.metrics import precision_recall_fscore_support

from DatasetsConsumers.EnronEvidence import EnronEvidence
from DatasetsConsumers.EnronFinancial import EnronFinancial
from DatasetsConsumers.Newsgroups import Newsgroups
from DatasetsConsumers.Spamassassin import Spamassassin
from DatasetsConsumers.Trustpilot import Trustpilot

datasets = [Newsgroups(), Spamassassin(), EnronEvidence(), EnronFinancial(), Trustpilot()]

for dataset in datasets:
    emails, labels = dataset.load(True)
    count = {0:90, 1:10}
    correct_preds = max(count.values())
    incorrect_preds = sum(count.values()) - correct_preds
    num = (correct_preds) / (sum(count.values()))
    y_test = [0] * correct_preds + [i + 1 for i in range(len(count) - 1)] * incorrect_preds
    y_test = y_test[:(correct_preds + incorrect_preds)]
    predictions = [0] * sum(count.values())
    precision, recall, _fscore, support = precision_recall_fscore_support(y_test, predictions)

    print(dataset.get_name())
    print('Class count:', count.values())
    print('Accuracy:', num)
    print('Average F-Score:', sum(_fscore)/len(_fscore))
