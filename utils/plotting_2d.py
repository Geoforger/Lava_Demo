import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lava.lib.dl.slayer as slayer


def calculate_pooling_dim(input_dim, kernel_dim, stride, order):
    """
    Function to calculate the size of a dimension after pooling

    Parameters
    ----------

    input_dim (int): Size of the dimension that is being pooled
    kernel_dim (int): Size of the pooling kernel along the same dimension
    stride (int): Stride of the pooling operation
    order (int): Number of pooling operations being performed
    """
    return int(((input_dim - kernel_dim) / stride) + (1 * order))


def heatmap_lava(data, orig_res, pooling_kernel, stride, order):
    """
    Function to plot lava events as a 2d heatmap of a given resolution

    Parameters
    ----------

    data (numpy array): Numpy array containing n timesteps of output data after pooling
    orig_res (tuple): Shape of the original data
    pooling_kernel (tuple): Shape of the pooling kernel used 
    stride (int): Stride of the pooling operation
    order (int): Number of pooling operations performed
    """
    output_y = calculate_pooling_dim(orig_res[0], pooling_kernel[0], stride, order)
    output_x = calculate_pooling_dim(orig_res[1], pooling_kernel[1], stride, order)
    output_res = (output_y, output_x)
    
    neurons, num_ts = data.shape
    heat_map = np.zeros(output_res)

    # x = ts
    # y = neuron ID
    flat_data = np.zeros(neurons)

    for neuron in range(neurons):
        flat_data[neuron] = np.count_nonzero(data[neuron])

    heat_map = np.reshape(flat_data, output_res)
    ax = sns.heatmap(heat_map, linewidth=0.0)
    
    return ax


def raster_plot(data, display=False, save=False, path=""):
    """
    Function to plot lava events as a raster plot

    Parameters
    ----------

    data (numpy array): Numpy array containing n timesteps of data
    """
    spikes_data = data.flatten()
    neuron_idx = 0

    for spike_train in spikes_data:
        if spike_train != []:
            y = np.ones_like(spike_train) * neuron_idx
            plt.plot(spike_train, y, 'k|', markersize=0.7)
        neuron_idx +=1
        
    plt.ylim = (0, len(spikes_data))
    plt.ylabel("Neuro Idx")
    plt.xlabel("Time (ms)")
    plt.title("Raster Plot")
    plt.setp(plt.gca().get_xticklabels(), visible=True)

    if display:
        plt.show()
    if save:    
        plt.savefig(path)