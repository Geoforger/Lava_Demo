import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.animation import FuncAnimation
from data_processor import DataProcessor
import torch  # Used to create torch tensor for integration with lava
import lava.lib.dl.slayer as slayer  # Used for creating the Events object


class AnimatedScatter(QWidget):
    def __init__(self, data):
        super().__init__()
        self.data = data

        # Set up the figure and axis for the plot
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        # Add canvas to the layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Initialize the scatter plot
        self.scatter = self.ax.scatter([], [])
        self.ax.set_xlim(0, data.shape[1])
        self.ax.set_ylim(0, data.shape[0])
        self.ax.set_xlabel("Timestep")
        self.ax.set_ylabel("Neuron Idx")

        # Prepare data for animation
        self.animation_data = self.prepare_animation_data()

        # Start the animation
        self.anim = FuncAnimation(
            self.figure,
            self.update_plot,
            frames=data.shape[1],
            interval=100,
            repeat=False,
        )

    def prepare_animation_data(self):
        # Pre-calculate the data for each frame
        animation_data = []
        for frame in range(self.data.shape[1]):
            y, x = np.where(self.data[:, : frame + 1] == 1)
            animation_data.append((x, y))
        return animation_data

    def update_plot(self, frame):
        # Update the scatter plot for the current frame
        x, y = self.animation_data[frame]
        self.scatter.set_offsets(np.column_stack([x, y]))
        return (self.scatter,)


def find_final_ts(data):
    temp_array = data.reshape(-1)
    max_ts = 0

    for element in temp_array:
        if element != []:
            max_element = max(element)
            if max_element > max_ts:
                max_ts = max_element

    return max_ts


def main():
    #
    filename = "/media/farscope2/T7/PhD/FrankaDatasets/Natural-12tex-100samples-linear_acceleration-vel-10-acc-20-1698856033.5525281/0-1-5.npy"
    data_proc = DataProcessor.load_data_np(path=filename)

    x_size = 640
    y_size = 480
    num_neurons = data_proc.data.shape[0] * data_proc.data.shape[1]
    num_timesteps = find_final_ts(data_proc.data)

    # Do some processing
    data_proc.offset_values(1)
    data = data_proc.create_lava_array(num_timesteps)

    # Sample data: replace this with your actual data

    app = QApplication(sys.argv)
    mainWin = AnimatedScatter(data)
    mainWin.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
