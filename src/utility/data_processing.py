import csv
import re
import statistics

import matplotlib.pyplot as plt
import numpy
import numpy as np

# file_name = ROOTPATH + "/Results/MLP_Tensorflow/Newsgroups/resultsfile.csv"

#file_name = 'C:\\Users\\Jens\\Documents\\Results\\2000\\{}\\{}\\resultsfile.csv'
file_name = 'C:\\Users\\Mads\\IdeaProjects\\Results\\2000\\{}\\{}\\resultsfile.csv'



def plot_line_graph(ax, data, title, xlabel, ylabel):
    for xdata, ydata, label in data:
        # ax.plot(xdata, ydata)
        trend = numpy.polyfit(xdata, ydata, 5)
        trendpoly = np.poly1d(trend)
        plt.plot(xdata, trendpoly(xdata), label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # plt.ylim([0, 1])
    # plt.yticks([i/10 for i in range(10)])


def plot_3d(data):
    X, Y, Z = data
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    fig = plt.subplots()
    ax = plt.axes(projection='3d')
    ax.scatter(X, Y, Z, c=Z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # ax.figure.colorbar()
    # ax.set_zlabel('z')
    plt.show()


def plot_bar_chart(bars, groups, title, xlabel, ylabel):
    n_groups = len(groups)

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.9 / n_groups

    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    c = 0

    for label, data in bars:
        _data, std_dev = data
        ax.bar(index + bar_width * c, _data, bar_width,
               alpha=0.5, color='blue' if label == 'LeakyReLU' else None,
               yerr=std_dev, error_kw=error_config,
               label=label)
        c += 1

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(groups)
    ax.legend()

    fig.tight_layout()
    plt.show()
    fig.savefig(title)


def parse_file(file_name):
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        data = []
        headers = []
        for row in csv_reader:
            if line_count == 0:
                headers = row
                # print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                data.append(row)
                line_count += 1
        return data, headers


def get_row_data(data, headers, row_idx, includeRelu=False):
    data_list = []
    for row in data:
        if not includeRelu and re.search('relu\)$', row[8]) is not None:
            continue
        data_list.append(row[row_idx])
    return data_list, headers[row_idx]


def process_bar_data(row_idx, dataset_name, algo):

    data, headers = parse_file(file_name.format(algo, dataset_name))

    # print(f'Processed {line_count} lines.')
    # steps = 100
    fscore_dict = {}
    re_list = []
    groups = []
    data.sort()
    print("{} {} {}".format(len(data), dataset_name, algo))
    top_10 = int(len(data) / 10)
    data = list(reversed(data))[:top_10]
    print('lowest: {} {}'.format(data[top_10 - 1][0], dataset_name))
    all_fscores = []
    for row in data:
        key = row[row_idx]
        # key = str(key.count("Dropout") - key.count("Dropout;0.0"))
        m = re.search('relu\)$', row[8])  # No relu output
        print(row[8])
        if m is not None:
            continue
        m = re.search('RNN(;\d+\.\d+,|., \d+, \')(\w+)', key)  # first dropout layer
        if m is None:
            print(m)
            print(key)
        key = m.group(2)
        if key == 'linear':
            key = 'LeakyReLU'
        # m = re.search('^.+?Dropout;([\d\.]+)', key)  # first dropout layer
        # m = re.search('(\w+)\)$', key) #Output function
        # m = re.search('^.+?(\w+)\)', key)
        # key = m.group(1)
        fscore = float(row[0].replace(',', '.'))
        all_fscores.append(fscore)
        val_list = fscore_dict.get(key, [])
        val_list.append(fscore)
        fscore_dict[key] = val_list

    if (fscore_dict.get('relu') is None):
        fscore_dict['relu'] = [0]
    if (fscore_dict.get('softmax') is None):
        fscore_dict['softmax'] = [0]
    if (fscore_dict.get('LeakyReLU') is None):
        fscore_dict['LeakyReLU'] = [0]
        '''
    if fscore_dict.get('0') is None:
        fscore_dict['0'] = [0]
    if fscore_dict.get('1') is None:
        fscore_dict['1'] = [0]
    if fscore_dict.get('2') is None:
        fscore_dict['2'] = [0]
    if fscore_dict.get('3') is None:
        fscore_dict['3'] = [0]'''
    for key in fscore_dict:
        val_array = fscore_dict[key]
        # avg_fscore = sum(val_array) / len(val_array)
        groups.append(key)
        if len(val_array) > 1:
            std = statistics.stdev(val_array)
        else:
            std = 0
        # std = 0
        mean_fscore = statistics.mean(val_array)
        # print(mean_fscore, std)
        re_list.append((key, mean_fscore, std))

        # print('{} {:.3f}'.format(dataset_name, max(all_fscores)))
    return re_list, headers[row_idx]


def run_bars():
    for algo in ['RNN_Tensorflow']:  # 'Bi_LSTM_Tensorflow', 'MLP_Tensorflow', 'RNN_Tensorflow']:
        # print(algo)
        datas = ['EnronFinancial', 'Spamassassin', 'Newsgroups', 'EnronEvidence', 'Trustpilot']

        data_grab = [process_bar_data(8, dataset, algo) for dataset in datas]

        # Returns (parameter, fscore) for each dataset
        title_value = ''
        bar_data = []
        f_score_lists = {}
        for i, tuple_list in enumerate(data_grab):
            dataset = datas[i]
            tuple, title_value = tuple_list
            for var_name, f_score, std_dev in tuple:
                list1, list2 = f_score_lists.get(var_name, ([], []))
                list1.append(f_score)
                list2.append(std_dev)
                f_score_lists[var_name] = (list1, list2)
        # print(f_score_lists)
        for key in f_score_lists:
            bar_data.append((key, f_score_lists[key]))

        # print(bar_data)
        bar_data.sort()
        plot_bar_chart(bar_data, datas, algo + ' activation function', 'Dataset', 'F-Score')


run_bars()

def process_line_data(row_idx, dataset_name, algo, ax):

    data, headers = parse_file(file_name.format(algo, dataset_name))

    fscore_dict = {}
    re_list = []
    groups = []
    data.sort()
    top_10 = int(len(data))
    data = list(reversed(data))[:top_10]
    print('lowest: {} {}'.format(data[top_10 - 1][0], dataset_name))
    all_fscores = []
    all_keys1 = []
    all_keys2 = []
    key_dict1 = {}
    key_dict2 = {}
    # print(int(data[0][6]))
    for row in data:
        key1 = float(row[row_idx])
        fscore = float(row[0].replace(',', '.'))
        key_list = key_dict1.get(key1, [])
        key_list.append(fscore)
        key_dict1[key1] = key_list

        '''
        m = re.findall('Bi_LSTM;(\d+)', key1)

        key1 = int(float(m[0]) / 2) * 2
        key2 = int(float(m[1]) / 2) * 2
        key_list = key_dict1.get(key1, [])
        key_list.append(fscore)
        key_dict1[key1] = key_list

        key_list = key_dict2.get(key2, [])
        key_list.append(fscore)
        key_dict2[key2] = key_list
        # all_fscores.append(fscore)
    for key in key_dict1:
        all_keys1.append(key)
        all_fscores.append(statistics.mean(key_dict1[key]))
    for key in key_dict2:
        all_keys2.append(key)
    #    all_fscores.append(statistics.mean(key_dict2[key]))
    '''

    for key in key_dict1:
        all_keys1.append(key)
        all_fscores.append(statistics.mean(key_dict1[key]))
    comb = list(zip(all_keys1, all_fscores))
    comb.sort()
    all_keys1, all_fscores1 = zip(*comb)
    '''
    comb1 = list(zip(all_keys2, all_fscores))
    comb1.sort()
    all_keys2, all_fscores2 = zip(*comb1)'''
    # plot_3d((all_keys1, all_keys2, all_fscores))
    plot_line_graph(ax, [(all_keys1, all_fscores, dataset_name)], 'Learning rate ' + dataset_name, 'Learning Rate',
                    'F-Score')


def run_line_graph():
    for algo in ['Bi_LSTM_Tensorflow', 'MLP_Tensorflow', 'RNN_Tensorflow']:
        # print(algo)
        datas = ['EnronFinancial', 'Spamassassin', 'Newsgroups', 'EnronEvidence', 'Trustpilot']

        fig, ax = plt.subplots()
        data_grab = [process_line_data(5, dataset, algo, ax) for dataset in datas]
        ax.set_title('Learning rate ' + algo)
        ax.legend()
        fig.tight_layout()
        plt.show()
        fig.savefig('Learning rate ' + algo)


def plot_line_data(ax, xdata, ydata, label=None):
    ax.plot(xdata, ydata, label=label)


def plot_trend_line(xdata, ydata, label=None):
    trend = numpy.polyfit(xdata, ydata, 5)
    trendpoly = np.poly1d(trend)
    plt.plot(xdata, trendpoly(xdata), label=label)


def sort_data(xdata, ydata):
    comb = list(zip(xdata, ydata))
    comb.sort()
    return zip(*comb)


def plot_all_hidden_dim_lines():
    for algo in ['RNN_Tensorflow']:  # ''Bi_LSTM_Tensorflow', 'MLP_Tensorflow', 'RNN_Tensorflow']:
        # print(algo)
        fig, ax = plt.subplots()
        for dataset_name in ['EnronFinancial', 'Spamassassin', 'Newsgroups', 'EnronEvidence', 'Trustpilot']:
            file_data, file_headers = parse_file(file_name.format(algo, dataset_name))
            use_relu = False

            all_f_scores, ylabel = get_row_data(file_data, file_headers, 0, use_relu)
            all_f_scores = [float(x.replace(',', '.')) for x in all_f_scores]
            min_f = min(all_f_scores)
            max_f = max(all_f_scores)
            all_f_scores_normed = [(x - min_f) / (max_f - min_f) for x in all_f_scores]

            all_hidden_layers, xlabel = get_row_data(file_data, file_headers, 8, use_relu)
            all_output_dims, label = get_row_data(file_data, file_headers, 6, use_relu)
            all_first_hidden_dim = [int(re.search('RNN;(\d+)', key).group(1)) for key in all_hidden_layers]
            all_first_hidden_dim = [dim / int(all_output_dims[i]) for i, dim in enumerate(all_first_hidden_dim)]
            xdata, ydata = sort_data(all_first_hidden_dim, all_f_scores_normed)
            # plot_line_data(ax, xdata, ydata, dataset_name)
            min_f = min(ydata)
            max_f = max(ydata)
            plot_trend_line(xdata, ydata, dataset_name)
            # plot_line_data(ax, xdata, ydata, dataset_name)
            # data_grab = [process_line_data(8, dataset, algo, ax) for dataset in datas]
        # ax.set_xticks([i for i in range(0, 50, 5)])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(algo + ': Hidden dim / Output dim, normalized')
        ax.legend()
        fig.tight_layout()
        plt.show()
        fig.savefig(algo + ': Hidden dim - Output dim, normalized')


def plot_layer_correlation():
    for algo in ['MLP_Tensorflow']:  # ''Bi_LSTM_Tensorflow', 'MLP_Tensorflow', 'RNN_Tensorflow']:
        # print(algo)
        fig, ax = plt.subplots()
        for dataset_name in ['EnronFinancial', 'Spamassassin', 'Newsgroups', 'EnronEvidence', 'Trustpilot']:
            print(file_name.format(algo, dataset_name))
            file_data, file_headers = parse_file(file_name.format(algo, dataset_name))
            use_relu = True

            all_f_scores, ylabel = get_row_data(file_data, file_headers, 0, use_relu)
            all_f_scores = [float(x.replace(',', '.')) for x in all_f_scores]
            min_f = min(all_f_scores)
            max_f = max(all_f_scores)
            all_f_scores = [(x - min_f) / (max_f - min_f) for x in all_f_scores]
            all_first_hidden_layers, xlabel = get_row_data(file_data, file_headers, 8, use_relu)
            all_first_hidden_dim = [int(re.findall('Dense;(\d+)', key)[0]) for key in all_first_hidden_layers]

            # all_output_dims, label = get_row_data(file_data, file_headers, 6, use_relu)
            # all_over_output_dim = [dim / int(all_output_dims[i]) for i, dim in enumerate(all_first_hidden_dim)]

            all_second_hidden_dim = [int(re.findall('Dense;(\d+)', key)[1]) for key in all_first_hidden_layers]
            all_dim_corr = [x / all_second_hidden_dim[i] for i, x in enumerate(all_first_hidden_dim)]
            xdata, ydata = sort_data(all_dim_corr, all_f_scores)
            xdata = xdata[:50]
            ydata = ydata[:50]
            # plot_line_data(ax, xdata, ydata, dataset_name)
            min_f = min(ydata)
            max_f = max(ydata)
            plot_trend_line(xdata, ydata, dataset_name)
            # plot_line_data(ax, xdata, ydata, dataset_name)
            # data_grab = [process_line_data(8, dataset, algo, ax) for dataset in datas]
        # ax.set_xticks([i for i in range(0, 50, 5)])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale('log')
        title = algo + ' Layer 1 dim / Layer 2 dim, normalized'
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        plt.show()
        fig.savefig(title.replace('/', ''))


# plot_layer_correlation()

'''
for algo in ['Bi_LSTM_Tensorflow', 'MLP_Tensorflow', 'RNN_Tensorflow']:
    # print(algo)
    fig, ax = plt.subplots()
    for dataset_name in ['EnronFinancial', 'Spamassassin', 'Newsgroups', 'EnronEvidence', 'Trustpilot']:
        data, headers = parse_file(file_name.format(algo, dataset_name)
        # print(file_name.format(algo, dataset_name)
        for i, row in enumerate(data):
            matches = len(re.findall('Dropout;0.0', row[8]))
            if matches == 3:
                print(file_name.format(algo, dataset_name), i)'''


def plot_optimizers():
    for algo in ['Bi_LSTM_Tensorflow', 'MLP_Tensorflow', 'RNN_Tensorflow']:
        # print(algo)
        fig, ax = plt.subplots()
        for dataset_name in ['EnronFinancial', 'Spamassassin', 'Newsgroups', 'EnronEvidence', 'Trustpilot']:
            print(file_name.format(algo, dataset_name))
            file_data, file_headers = parse_file(file_name.format(algo, dataset_name))
            use_relu = True

            all_f_scores, ylabel = get_row_data(file_data, file_headers, 0, use_relu)
            all_f_scores = [float(x.replace(',', '.')) for x in all_f_scores]

            all_optimizers, xlabel = get_row_data(file_data, file_headers, 4, use_relu)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        title = algo + ' Optimizers'
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        plt.show()
        fig.savefig(title.replace('/', ''))


def get_max_fscores():
    for algo in ['Bi_LSTM_Tensorflow', 'MLP_Tensorflow', 'RNN_Tensorflow']:
        print(algo)
        fig, ax = plt.subplots()
        for dataset_name in ['EnronFinancial', 'Spamassassin', 'Newsgroups', 'EnronEvidence', 'Trustpilot']:
            file_data, file_headers = parse_file(file_name.format(algo, dataset_name))
            all_f_scores, ylabel = get_row_data(file_data, file_headers, 0, True)
            all_f_scores = [float(x.replace(',', '.')) for x in all_f_scores]
            print("{} {:.3f}".format(dataset_name, max(all_f_scores)))


def plot_dropout():
    for algo in ['Bi_LSTM_Tensorflow']:  # , 'MLP_Tensorflow', 'RNN_Tensorflow']:
        # print(algo)
        fig, ax = plt.subplots()
        for dataset_name in ['EnronFinancial']:  # , 'Spamassassin', 'Newsgroups', 'EnronEvidence', 'Trustpilot']:
            print(file_name.format(algo, dataset_name))
            file_data, file_headers = parse_file(file_name.format(algo, dataset_name))
            use_relu = False

            all_f_scores, ylabel = get_row_data(file_data, file_headers, 0, use_relu)
            all_f_scores = [float(x.replace(',', '.')) for x in all_f_scores]
            print(max(all_f_scores))
            all_hidden, xlabel = get_row_data(file_data, file_headers, 8, use_relu)
            first_dropout = [float(re.findall('Dropout(;|\', )(\d+\.\d+)', x)[0][1]) + float(
                re.findall('Dropout(;|\', )(\d+\.\d+)', x)[1][1]) + float(
                re.findall('Dropout(;|\', )(\d+\.\d+)', x)[2][1]) for x in all_hidden]
            all_f_scores, first_dropout = sort_data(all_f_scores, first_dropout)
            all_f_scores = all_f_scores
            first_dropout = first_dropout
            drop_dict = {}
            for i, x in enumerate(first_dropout):
                dropout_list = drop_dict.get(x, [])
                dropout_list.append(all_f_scores[i])
                drop_dict[x] = dropout_list
            xdata = list(drop_dict.keys())
            xdata.sort()
            ydata = [statistics.mean(drop_dict[val]) for val in xdata]
            plot_line_data(ax, xdata, ydata, dataset_name)

        ax.set_xlabel('Dropout')
        ax.set_ylabel(ylabel)
        title = algo + ' Dropout first layer'
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        plt.show()
        fig.savefig(title.replace('/', ''))


# plot_dropout()

def plot_normalized_dropout():
    for algo in [
        # 'Bi_LSTM_Tensorflow',
        'MLP_Tensorflow',
        # 'RNN_Tensorflow'
    ]:
        # print(algo)
        fig, ax = plt.subplots()
        for dataset_name in ['EnronFinancial', 'Spamassassin', 'Newsgroups', 'EnronEvidence', 'Trustpilot']:
            print(file_name.format(algo, dataset_name))
            file_data, file_headers = parse_file(file_name.format(algo, dataset_name))
            use_relu = False
            file_data.sort()
            top_10 = int(len(file_data) / 10)
            file_data = list(reversed(file_data))[:top_10]
            all_f_scores, ylabel = get_row_data(file_data, file_headers, 0, use_relu)
            all_f_scores = [float(x.replace(',', '.')) for x in all_f_scores]

            # all_f_scores = [(x-minv)/(maxv-minv) for x in all_f_scores]
            # maxvnew = max(all_f_scores)
            # minvnew = min(all_f_scores)
            print(max(all_f_scores))
            all_hidden, xlabel = get_row_data(file_data, file_headers, 8, use_relu)
            if algo == "MLP_Tensorflow":
                first_dropout = [float(re.findall('Dropout(;|\', )(\d+\.\d+)', x)[0][1]) for x in all_hidden]
            else:
                first_dropout = [float(re.findall('Dropout(;|\', )(\d+\.\d+)', x)[0][1])
                                 + float(
                    re.findall('Dropout(;|\', )(\d+\.\d+)', x)[1][1]) + float(
                    re.findall('Dropout(;|\', )(\d+\.\d+)', x)[2][1]) for x in all_hidden]
            all_f_scores, first_dropout = sort_data(all_f_scores, first_dropout)
            all_f_scores = all_f_scores
            first_dropout = first_dropout
            drop_dict = {}
            for i, x in enumerate(first_dropout):
                dropout_list = drop_dict.get(x, [])
                dropout_list.append(all_f_scores[i])
                drop_dict[x] = dropout_list
            xdata = list(drop_dict.keys())
            xdata.sort()
            ydata = [statistics.mean(drop_dict[val]) for val in xdata]
            maxv = max(ydata)
            minv = min(ydata)
            ydata = [(x - minv) / (maxv - minv) for x in ydata]
            plot_trend_line(xdata, ydata, dataset_name)

        ax.set_xlabel('Summed dropout')
        ax.set_ylabel('Normalized fscore')
        title = algo + ' Dropout sum layer'
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        plt.show()
        fig.savefig(title.replace('/', ''))

# plot_normalized_dropout()
