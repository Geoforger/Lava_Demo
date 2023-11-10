from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
import numpy as np

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel

from IPython.display import display, clear_output
import matplotlib.pyplot as plt

from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg


class Visualiser(AbstractProcess):
    """Visualiser process for spike input or output.

    This process plots a graph of the input in real time

    Parameters
    ----------

    in_shape (int): Shape of input vector.
    """

    def __init__(self, in_shape, sample_length, window_size):
        super().__init__()
        # Set process variables
        self.a_in = InPort(shape=in_shape)
        # self.sample_length = Var(shape=(1,), init=sample_length)
        self.proc_params["in_shape"] = in_shape
        self.proc_params["sample_length"] = sample_length
        self.proc_params["window_size"] = window_size


# @implements(proc=Visualiser, protocol=LoihiProtocol)
# @requires(CPU)
# class VisualiserModel(PyLoihiProcessModel):
#     a_in = LavaPyType(PyInPort.VEC_DENSE, float)
#     # sample_length: int = LavaPyType(int, int)

#     def __init__(self, proc_params=None) -> None:
#         super().__init__()
#         self.fig = plt.figure(figsize=(10, 5))
#         self.ax1 = self.fig.add_subplot()  # 1, 1, 1

#         self.sample_length = proc_params["sample_length"]
#         self.window_size = proc_params["window_size"]
#         in_shape = proc_params["in_shape"]

#         # Create empty sample array
#         self.data = np.zeros((in_shape[0], self.sample_length))

#         print(
#             f"Setup visualiser. Sample Length: {self.sample_length}, Input shape: {in_shape}"
#         )

#     def run_spk(self):
#         # Recieve a vector of 0,1s at input
#         data_in = self.a_in.recv()

#         # Add the vector to the array
#         self.data[:, (self.time_step - 1)] = data_in
#         points = np.nonzero(self.data)

#         # Clear plot and replot data
#         self.ax1.clear()
#         self.ax1.scatter(points[1], points[0], s=1.0)

#         if self.time_step < self.window_size:
#             self.ax1.set_xlim(0, self.window_size)
#         else:
#             self.ax1.set_xlim(
#                 self.time_step - self.window_size, self.time_step + self.window_size
#             )

#         # self.ax1.set_yticks(np.arange(self.a_in.shape[0]))
#         self.ax1.set_ylabel("Neuron Idx")
#         self.ax1.set_xlabel("Time Step")
#         self.ax1.set_yticks(np.arange(0, self.a_in.shape[0], 1))

#         # Live display
#         clear_output(wait=True)
#         display(self.fig)


@implements(proc=Visualiser, protocol=LoihiProtocol)
@requires(CPU)
# @tag("pyqt")
class PyQtVisualiserModel(PyLoihiProcessModel):
    a_in = LavaPyType(PyInPort.VEC_DENSE, float)

    def __init__(self, proc_params=None) -> None:
        super().__init__()

        # Create pyqt window / figure
        self.app = QtWidgets.QApplication([])
        self.plt = pg.plot(title="Dynamic Plotting with PyQtGraph")
        # self.scatter = pg.ScatterPlotItem(size=10)

        # Set process parameters
        self.sample_length = proc_params["sample_length"]
        self.window_size = proc_params["window_size"]
        self.half_window = int(self.window_size / 2)
        in_shape = proc_params["in_shape"]

        # Create empty sample array
        self.data = np.zeros((in_shape[0], self.sample_length))

        print(
            f"Setup PyQt visualiser. Sample Length: {self.sample_length}, Input shape: {in_shape}"
        )

    def run_spk(self):
        data_in = self.a_in.recv()

        # Add the vector to the array
        self.data[:, (self.time_step - 1)] = data_in

        if self.time_step < self.window_size:
            plot_section = self.data[:, 0 : (self.time_step - 1)]
        else:
            plot_section = self.data[
                :, (self.time_step - self.half_window) : self.time_step
            ]

        points_y, points_x = np.nonzero(plot_section)

        if points_y.size != 0:
            # Randomly select indices for subsample
            subsample_size = int(points_y.size * 0.2)

            subsample_indices = np.random.choice(
                points_y.size, size=subsample_size, replace=False
            )

            # Extract subsample from the data
            points_x = points_x[subsample_indices]
            points_y = points_y[subsample_indices]

            # # Create scatter points
            self.plt.clear()
            self.plt.plot(points_x, points_y, pen=None, symbol="o", size=0.5)
            self.plt.setYRange(0, self.a_in.shape[0], padding=0)

            # TODO: Fix
            # if self.time_step < self.window_size:
            #     # self.plt.plotItem.vb.setLimits(
            #     #     xMin=0,
            #     #     xMax=self.window_size,
            #     #     yMin=0,
            #     #     yMax=self.a_in.shape[0],
            #     # )
            #     ticks = np.arange(self.time_step)
            # else:
            #     ticks = np.arange(self.time_step - self.window_size, self.time_step, 10)

            # ay = self.plt.getAxis("bottom")
            # ay.setTicks([[(v, str(v)) for v in enumerate(list(ticks))]])

            # Update plot?
            self.app.processEvents()

        if self.time_step == self.sample_length:
            pg.QtWidgets.QApplication.exec_()
