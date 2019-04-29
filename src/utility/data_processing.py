import csv
import matplotlib.pyplot as plt

# file_name = ROOTPATH + "/Results/MLP_Tensorflow/Newsgroups/resultsfile.csv"
file_name = "C:\\Users\\Jens\\Downloads\\resultsfile med lossfunction header.csv"
with open(file_name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    data = []
    headers = []
    for row in csv_reader:
        if line_count == 0:
            headers=row
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            data.append(row)
            line_count += 1
    print(f'Processed {line_count} lines.')
    steps = 100
    fscore_dict = {}
    for row in data:
        idx = (int(float(row[0]) * steps))
        input_func = row[2].strip()
        list = fscore_dict.get(input_func)
        if list is None:
            fscore_dict[input_func] = [0] * steps
        if fscore_dict[input_func][idx] is None:
            fscore_dict[input_func][idx] = 0
        fscore_dict[input_func][idx] += 1

    # fig = plt.figure(figsize=(10, 10))
    plt.title(headers[2])
    plt.xlabel('FScore')
    plt.ylabel('Amount')
    for func in fscore_dict:
        plt.plot(fscore_dict[func], label=func, solid_capstyle='round')
    plt.legend()
    plt.show()
