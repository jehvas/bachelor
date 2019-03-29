# Compute Term Frequency
"""
Term Frequency (tf): gives us the frequency of the word in each document in the corpus.
It is the ratio of number of times the word appears in a document compared to the total number of words in that document.
It increases as the number of occurrences of that word within the document increases. Each document has its own tf.
"""


def compute_tf(word_dicts, emails):
    all_tfs = []
    for i in range(len(emails)):
        email = emails[i]
        word_dict = word_dicts[i]
        tf_dict = {}
        email_count = len(email)
        for word, count in word_dict.items():
            tf_dict[word] = count / float(email_count)
        all_tfs.append(tf_dict)
    return all_tfs


# Compute Inverse Data Frequency
"""
Inverse Data Frequency (idf): used to calculate the weight of rare words across all documents in the corpus.
The words that occur rarely in the corpus have a high IDF score. It is given by the equation below.
"""


def compute_idf(word_count_list, doc_list):
    import math
    n = len(doc_list)
    idf_dict = {}
    for occurrence_dict in word_count_list:
        for word, val in occurrence_dict.items():
            if val > 0:
                idf_dict[word] = idf_dict.get(word, 0) + 1
            else:
                print("What")

    for word, val in idf_dict.items():
        idf_dict[word] = math.log10(n / float(val))

    return idf_dict


# Compute TF*IDF
def compute_tfidf(word_count_list, emails):
    tf =compute_tf(word_count_list, emails)
    idf = compute_idf(word_count_list, emails)
    all_tfidf = tf
    for email_tfs in all_tfidf:
        for word in email_tfs:
            email_tfs[word] = email_tfs[word] * idf[word]
            if email_tfs[word] < 0.0005 and word != 'subject':
                print(word)
    return all_tfidf
