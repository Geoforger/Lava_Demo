from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
import numpy as np

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel

from IPython.display import display, clear_output
import matplotlib.pyplot as plt

class DecisionMaker(AbstractProcess):
    """Decision making process for real-time classification.

    By accumulating data for each input neuron in the input vector, this process outputs a classification in real-time.
    The offset and threshold params are used to tune the accuracy of the classification. Increases to either values increases accuracy at the expense of time.
        > Higher offset values will increase the time required for a classification.
        > Threshold values should be tuned based on your offset.

    Parameters
    ----------

    in_shape (int): Shape of input vector.
    offset (OPTIONAL. Default=0) (int): Optional param for total information required to output a decision.
    threshold (OPTIONAL. Default=0) (float): Optional param for confidence in the gathered information required to output a decision.
    prior (OPTIONAL. Default=0) (int): Optional param to set the prior of each information accumulator. Low values increase sensitivity at the cost of instability.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("in_shape")
        prior = kwargs.get("prior", 0)
        
        self.proc_params["offset"] = kwargs.get("offset", 0)
        self.proc_params["threshold"] = kwargs.get("threshold", 0)

        # Set process variables
        self.shape = shape[-1] # This can be passed as an integer
        self.accumulator = Var(shape=shape, init=(np.ones(self.shape) * prior))
        self.confidence = Var(shape=(1,), init=0)

        self.a_in = InPort(shape=shape)
        self.out = OutPort(shape=(1,))


@implements(proc=DecisionMaker, protocol=LoihiProtocol)
@requires(CPU)
# @tag("floating_pt")
class PyDecisionMaker(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.uint32)
    accumulator: np.ndarray = LavaPyType(np.ndarray, int)
    # NOTE: THIS IS SO STUPID. I CAN ONLY PROBE THIS VAR IF ITS AN ARRAY
    confidence: np.ndarray = LavaPyType(np.ndarray, float)

    def __init__(self, proc_params=None):
        super().__init__()
        self.offset = proc_params["offset"]
        self.threshold = proc_params["threshold"]

    # Function to run on each timestep
    def run_spk(self):
        # Get input vector
        data_in = self.a_in.recv()
        # Accumulate input data
        self.accumulator = self.accumulator + data_in

        # Perform calculcations
        total_spikes = np.sum(self.accumulator)
        max_spikes = np.amax(self.accumulator)
        if total_spikes > 0:
            self.confidence = np.array([max_spikes / total_spikes])

            # Send output label if passing classification criteria. Else send an invalid classification
            if (total_spikes >= self.offset) and (self.confidence[0] >= self.threshold):
                # NOTE: This returns the first max value in the array if there are multiple instances of the same value present
                self.out.send(np.array([np.argmax(self.accumulator)]))
            else:
                self.out.send(np.array([16]))
        else:
            self.out.send(np.array([16]))


class DecisionMakerVisualiser(AbstractProcess):
    """Visualiser process for real-time classification.

    This process plots the status of the decision algorithm in real time

    Parameters
    ----------

    in_shape (int): Shape of input vector.
    """
    def __init__(self, in_shape):
            super().__init__()
            # Set process variables
            self.a_in = InPort(shape=in_shape)
            self.proc_params['in_shape'] = in_shape
    

@implements(proc=DecisionMakerVisualiser, protocol=LoihiProtocol)
@requires(CPU)
class DecisionMakerVisualiserModel(PyLoihiProcessModel): 
    a_in = LavaPyType(PyInPort.VEC_DENSE, float)
    # sample_length: int = LavaPyType(int, int)
    
    def __init__(self, proc_params=None) -> None:
        super().__init__()
        # Create subfigures
        self.fig, (self.bar, self.label) = plt.subplots(2,1, figsize=(10,7), gridspec_kw={'height_ratios': [4, 1]})

        in_shape = proc_params["in_shape"]
        
        # Create empty sample array
        self.data = np.zeros(in_shape[0])
        
        print(f"Setup decision maker visualiser. Input shape: {in_shape}")

    def run_spk(self):
        # Recieve a vector of 0,1s at input
        data_in = self.a_in.recv()
        
        # Add the vector to the array
        self.data = self.data + data_in
        
        # Find label and confidence
        label = np.argmax(self.data)
        confidence  = np.max(self.data) / np.sum(self.data)
        
        # Clear plot and replot data
        self.bar.clear()
        self.bar.bar(np.arange(self.a_in.shape[-1]), self.data)
        self.bar.set_xticks(np.arange(self.a_in.shape[-1]))
        # self.ax1.set_yticks(np.arange(self.a_in.shape[0]))
        self.bar.set_ylabel("Num Spikes")
        self.bar.set_xlabel("Output Label")
        
        # Add text readout of confidence and output label
        self.label.text(x=0.42, y=0.5, s=f"Output Classification: {label} \n Confidence: {confidence}", verticalalignment="center")
        self.label.axis('off')

        # Live display
        clear_output(wait=True)
        display(self.fig)






# Testing function
def main():
    # Test params
    sim_length = 100
    input_size = 11

    # Create input events
    import lava.lib.dl.slayer as slayer

    # Create random vector for input
    input = np.random.rand(input_size, sim_length)
    for y, x in np.ndindex(input.shape):
        if input[y][x] > 0.5:
            input[y][x] = 1
        else:
            input[y][x] = 0
    input = np.expand_dims(input, axis=0)
    proc_input = slayer.io.tensor_to_event(input, sampling_time=1)

    # Create components
    from lava.proc import io

    input_buffer = io.source.RingBuffer(
        data=proc_input.to_tensor(dim=(1, input_size, sim_length)).squeeze()
    )

    decision_maker = DecisionMaker(in_shape=input_size)
    output_buffer = io.sink.RingBuffer(shape=(1,), buffer=sim_length)

    # Connect components
    input_buffer.s_out.connect(decision_maker.a_in)
    decision_maker.out.connect(output_buffer.a_in)

    # Create probe for confidence
    from lava.proc.monitor.process import Monitor

    monitor_confidence = Monitor()
    monitor_confidence.probe(decision_maker.confidence, sim_length)

    # Run simulation
    from lava.magma.core.run_conditions import RunSteps
    from lava.magma.core.run_configs import Loihi1SimCfg

    run_config = Loihi1SimCfg()
    run_condition = RunSteps(num_steps=sim_length)

    print("Running sim")
    decision_maker.run(condition=run_condition, run_cfg=run_config)
    output = output_buffer.data.get()
    monitor_out = monitor_confidence.get_data()
    decision_maker.stop()
    print("Finished sim")

    # Print data output
    print(output)
    print(monitor_out)


if __name__ == "__main__":
    main()
