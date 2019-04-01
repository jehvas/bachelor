import matplotlib.pyplot as plt


def plot_data(data, x_label, y_label, title, dataset, algorithm, save=True, ticks = None):
    # visualization loss
    for plotTuple in data:
        x_data, y_data = plotTuple
        plt.plot(x_data, y_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim([0, 1])
    if ticks is not None:
        plt.yticks(ticks)
    filename = "{} - {}.  {} vs {} ".format(algorithm, dataset.get_name(), x_label, y_label)
    title = "{}".format(title)
    plt.title(filename + '\n' + title)
    plt.grid(True)
    if save:
        plt.savefig(filename + title + '.png')
    plt.show()
