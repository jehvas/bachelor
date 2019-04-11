import matplotlib.pyplot as plt


class PlotClass:
    data = None
    x_label = None
    y_label = None
    parameters = None
    dataset = None
    algorithm = None
    ticks = None
    legend = None

    def __init__(self, data, x_label, y_label, parameters, dataset, algorithm, ticks=None, legend=None):
        self.data = data
        self.x_label = x_label
        self.y_label = y_label
        self.parameters = parameters
        self.dataset = dataset
        self.algorithm = algorithm
        self.ticks = ticks
        self.legend = legend


def plot_data(plot_class, save=True):
    # visualization loss
    for plotTuple in plot_class.data:
        x_data, y_data = plotTuple
        plt.plot(x_data, y_data)
    plt.xlabel(plot_class.x_label)
    plt.ylabel(plot_class.y_label)
    if plot_class.ticks is not None:
        plt.ylim([0, 1])
        plt.yticks(plot_class.ticks)
    title = "{} - {}. {} vs {} ".format(plot_class.algorithm, plot_class.dataset.get_name(), plot_class.x_label,
                                        plot_class.y_label)

    params_text = "\n".join("{}: {}".format(k, v) for k, v in plot_class.parameters.items())
    plt.title(title, fontsize=18)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(1.025, 1., params_text, fontsize=14, verticalalignment='top', bbox=props)
    plt.grid(True)
    # plt.subplots_adjust(right=0.8)
    if save:
        plt.savefig('imgs/' + title + '.png')
    if plot_class.legend is not None:
        plt.legend(plot_class.legend[0], loc=plot_class.legend[1])
    plt.show()
