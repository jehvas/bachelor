import csv

import math
import matplotlib.pyplot as plt

# file_name = ROOTPATH + "/Results/MLP_Tensorflow/Newsgroups/resultsfile.csv"
datas = ['EnronFinancial', 'Spamassassin', 'Newsgroups', 'EnronEvidence', 'Trustpilot']
for dataset in datas:
    file_name = "C:\\Users\\Jens\\IdeaProjects\\Bachelor\\Results\\RNN_Tensorflow\\"+dataset+"\\resultsfile.csv"
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        data = []
        headers = []
        for row in csv_reader:
            if line_count == 0:
                headers = row
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                data.append(row)
                line_count += 1
        print(f'Processed {line_count} lines.')
        steps = 100
        fscore_dict = {}
        row_idx = 0
        max_fscore = 0
        for row in data:
            max_fscore = max(max_fscore, float(row[row_idx].replace(',', '.')))
            '''
            idx = (int(float(row[0]) * steps))
            input_func = row[row_idx].strip()
            list = fscore_dict.get(input_func)
            if list is None:
                fscore_dict[input_func] = [0] * steps
            if fscore_dict[input_func][idx] is None:
                fscore_dict[input_func][idx] = 0
            fscore_dict[input_func][idx] += 1
            '''
        print(dataset, '{:.3f}'.format(max_fscore))
        '''
        # fig = plt.figure(figsize=(10, 10))
        plt.title(headers[row_idx])
        plt.xlabel('FScore')
        plt.ylabel('Amount')
        for func in fscore_dict:
            plt.plot(fscore_dict[func], label=func, solid_capstyle='round')
        plt.legend()
        plt.show()
        '''
