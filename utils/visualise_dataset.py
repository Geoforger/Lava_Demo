import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from plotting_2d import heatmap_lava, raster_plot
from data_processor import DataProcessor
from utils import time_from_accel
import glob

def main():
    # Import data
    path = "/home/george/Documents/FrankaLava/data/Natural-12tex-100samples-linear_velocity-vel-40-1698054720.2607744/"
    files = glob.glob(f"{path}/*.npy")
    data_proc = DataProcessor.load_data_np(path=files[1])

    # Meta params
    start_vel = 10
    distance = 60
    acceleration = 20
    dt = 0.01
    # sample_length = time_from_accel(start_vel, acceleration, distance) * 1000
    sample_length = 2000
    
    # Preprocess data
    data_proc.pixel_reduction(160, 170, 60, 110)
    # data_proc.offset_values(0, reduce=True) # Remove values lower than 0
    data_proc.remove_cuttoff(sample_length) # Remove events after the sample has ended

    print(f"Shape of data pre pooling: {data_proc.data.shape}")

    # Plot heatmap
    # fig, ax = plt.subplot(2)

    data_proc.plot_data(display=True, normalise=True)
    # plt.clf()

    # Try pooling
    kernel = (4,4)
    stride = 4
    threshold = 2
    data_proc.threshold_pooling(kernel, stride, threshold)
    print(f"Shape of data post pooling: {data_proc.data.shape}")

    # Plot heatmap
    data_proc.plot_data(display=True, normalise=True)
    # plt.clf()
    plt.show()

    # Plot raster plot
    raster_plot(data_proc.data, display=True)


if __name__ == "__main__":
    main()