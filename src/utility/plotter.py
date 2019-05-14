import matplotlib.pyplot as plt


class PlotClass:
    data = None
    x_label = None
    y_label = None
    dataset = None
    algorithm = None
    ticks = None
    legend = None

    def __init__(self, data, x_label, y_label, dataset_name, algorithm, ticks=None, legend=None):
        self.data = data
        self.x_label = x_label
        self.y_label = y_label
        self.dataset = dataset_name
        self.algorithm = algorithm
        self.ticks = ticks
        self.legend = legend


def plot_data(plot_class, save_file_path="", save=True, show=False):
    # visualization loss
    x_data, y_data = plot_class.data
    plt.plot(x_data, y_data)
    plt.xlabel(plot_class.x_label)
    plt.ylabel(plot_class.y_label)
    if plot_class.ticks is not None:
        plt.ylim([0, 1])
        plt.yticks(plot_class.ticks)
    title = "{} - {}. {} vs {} ".format(plot_class.algorithm, plot_class.dataset, plot_class.x_label,
                                        plot_class.y_label)

    plt.title(title, fontsize=18)
    plt.grid(True)
    if save_file_path != "":
        plt.savefig(save_file_path)
    elif save:
        plt.savefig('imgs/' + title + '.png')
    if plot_class.legend is not None:
        plt.legend(plot_class.legend[0], loc=plot_class.legend[1])
    if show:
        plt.show()
    plt.close("all")
