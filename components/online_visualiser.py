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
from lava.magma.core.process.variable import Var

from IPython.display import display, clear_output
import matplotlib.pyplot as plt


class OnlineVisualiser(AbstractProcess):
    """Visualiser process for spike input or output.

    This process plots a graph of the input in real time

    Parameters
    ----------

    in_shape (int): Shape of input vector.
    """
    def __init__(self, in_shape, sample_length, frame_shape):
        super().__init__()
        # Set process variables
        # self.fig = plt.figure(figsize=(15,5))
        self.a_in = InPort(shape=in_shape)
        # self.sample_length = Var(shape=(1,), init=sample_length)
        self.proc_params['in_shape'] = in_shape
        self.proc_params['sample_length'] = sample_length
        self.proc_params['frame_shape'] = frame_shape
        # self.frame_shape = frame_shape
        self.data = Var(shape=in_shape, init=np.zeros(in_shape, dtype=int))
        self.idx = Var(shape=in_shape, init=np.zeros(in_shape, dtype=int))

            
@implements(proc=OnlineVisualiser, protocol=LoihiProtocol)
@requires(CPU)
class OnlineVisualiserModel(PyLoihiProcessModel): 
    
    a_in = LavaPyType(PyInPort.VEC_SPARSE, int)
    data: np.ndarray = LavaPyType(np.ndarray, int)
    idx: np.ndarray = LavaPyType(np.ndarray, int)

    # sample_length: int = LavaPyType(int, int)
    
    def __init__(self, proc_params=None) -> None:
        super().__init__()
        self.fig = plt.figure(figsize=(24,18))
        self.ax1 = self.fig.add_subplot() #1, 1, 1
        
        self.sample_length = proc_params["sample_length"]
        # self.window_size = proc_params["window_size"]
        in_shape = proc_params["in_shape"]
        self.frame_shape = proc_params["frame_shape"]
        
        # Create empty sample array
        self.data = np.zeros((in_shape[0], self.sample_length))
        
        print(f"Setup visualiser. Sample Length: {self.sample_length}, Input shape: {in_shape}")

    def run_spk(self):
        # Recieve a vector of 0,1s at input
        data, idx = self.a_in.recv()
        idx = np.unravel_index(idx,self.frame_shape[::-1])[::-1]

        # self.data = np.pad(data,
        #                    pad_width=(0, self.in_port.shape[0] - data.shape[0]))
        # self.idx = np.pad(idx,
        #                   pad_width=(0, self.in_port.shape[0] - data.shape[0]))
        
        # Add the vector to the array
        # self.data[:,(self.time_step-1)] = data_in
        # points = np.nonzero(self.data)
        
        # Clear plot and replot data
        self.ax1.clear()
        self.ax1.scatter(idx[1], idx[0])
        
        # if self.time_step < self.window_size:
        #     self.ax1.set_xlim(0, self.window_size)
        # else:
        #     self.ax1.set_xlim(self.time_step-self.window_size, self.time_step + self.window_size)
            
        # self.ax1.set_yticks(np.arange(self.a_in.shape[0]))
        self.ax1.set_ylabel("y")
        self.ax1.set_xlabel("x")

        # Live display
        clear_output(wait=True)
        display(self.fig)
