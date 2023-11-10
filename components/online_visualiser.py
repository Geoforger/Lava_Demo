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
import time


plt.ion()

class DynamicUpdate():
    #Suppose we know the x range
    min_x = 0
    max_x = 240
    min_y = 0
    max_y = 180

    def on_launch(self):
        #Set up plot
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([],[], '.')
        #Autoscale on unknown axis and known lims on the other
        # self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(self.min_x, self.max_x)
        self.ax.set_ylim(self.min_y, self.max_y)
        #Other stuff
        # self.ax.grid()

    def on_running(self, xdata, ydata):
        #Update data (with the new _and_ the old points)
        self.lines.set_xdata(xdata)
        self.lines.set_ydata(ydata)
        # #Need both of these in order to rescale
        # self.ax.relim()
        # self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    #Example
    def __call__(self):
        self.on_launch()
        xdata = []
        ydata = []
        for x in np.arange(0,10,0.5):
            xdata.append(x)
            ydata.append(np.exp(-x**2)+10*np.exp(-(x-7)**2))
            self.on_running(xdata, ydata)
            time.sleep(0.01)
        return xdata, ydata

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

        self.dynamic_figure = DynamicUpdate()
        self.dynamic_figure.on_launch()
        
        self.sample_length = proc_params["sample_length"]
        in_shape = proc_params["in_shape"]
        self.frame_shape = proc_params["frame_shape"]
        
        # Create empty sample array
        self.data = np.zeros((in_shape[0], self.sample_length))
        
        print(f"Setup visualiser. Sample Length: {self.sample_length}, Input shape: {in_shape}")

    def run_spk(self):
        # Recieve a vector of 0,1s at input

        data, idx = self.a_in.recv()

        if len(data)>10:
            idx = np.unravel_index(idx,self.frame_shape[::-1])[::-1]     
            self.dynamic_figure.on_running(idx[0],idx[1])
        # time.sleep(0.001)


