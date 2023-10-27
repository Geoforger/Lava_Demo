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

class Visualiser(AbstractProcess):
    """Visualiser process for real-time classification.

    This process plots a graph of the input in real time

    Parameters
    ----------

    in_shape (int): Shape of input vector.
    """
    def __init__(self, in_shape, sample_length):
            super().__init__()
            # Set process variables
            self.a_in = InPort(shape=in_shape)
            self.sample_length = Var(shape=(1,), init=sample_length)
            
            
@implements(proc=Visualiser, protocol=LoihiProtocol)
@requires(CPU)
class VisualiserModel(PyLoihiProcessModel): 
    a_in = LavaPyType(PyInPort.VEC_DENSE, float)
    sample_length: int = LavaPyType(int, int)
    
    def __init__(self) -> None:
        super().__init__()
        self.fig = plt.figure(figsize=(15,5))
        self.ax1 = self.fig.add_subplot()
        
        # Create empty sample array
        self.data = np.zeros((self.a_in.shape[0], self.sample_length))

    def run_spk(self):
        # Recieve a vector of 0,1s at input
        data_in = self.a_in.recv()
        
        # Add the vector to the array
        self.data[:,self.time_step] = data_in
        points = np.nonzero(self.data)
        
        # Clear plot and replot data
        self.ax1.clear()
        self.ax1.scatter(points[1], points[0])
        self.ax1.set_xticks(np.arange(self.sample_length))
        self.ax1.set_yticks(np.arange(self.a_in.shape[0]))

        # Live display
        clear_output(wait=True)
        display(self.fig)
