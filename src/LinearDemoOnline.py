
import sys
sys.path.append(".")
from utils.data_gatherer_neurotac import DataCollector

# Import all the components of our network
from components.visualiser import Visualiser
# from utils.utils import calculate_pooling_dim

# Import lava related libraries
# import numpy as np
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc import io
from lava.lib.dl import netx#, slayer
from lava.proc.event_data.io.dv_stream import DvStream, DvStreamPM
from lava.magma.core.process.ports.ports import OutPort

# Import trained network model
net = netx.hdf5.Network(net_config="./networks/network.net")

# Create input ring buffer containing our input tensor we collected and preprocessed
# proc_params = {
#     "address":"127.0.0.1",
#     "port":53476,
#     "shape_out": net.inp.shape,
#     "shape_frame_in": net.inp.shape,
#     "seed_sub_sampling": 2
# }

source = DvStream(address="127.0.0.1",
    port=53476,
    shape_frame_in=(640,480),
    shape_out=net.inp.shape)

# Output ring buffer to contain output spikes
sample_length = 3000
sink = io.sink.RingBuffer(shape=(2,), buffer=sample_length)
# Create a visualiser object that will let us see network output in real-time
vis = Visualiser(in_shape=net.out.shape, sample_length=sample_length, window_size=50)

# Connect processes
# source.out_port = OutPort(shape=proc_params["shape_out"]) 
source.out_port.connect(net.inp)
net.out.connect(sink.a_in)
net.out.connect(vis.a_in)

# Run the network for long enough to get all of the data through the network
run_condition = RunSteps(num_steps=1)
# RunContinuous()

# Map the defined encoder and adapter to their proc models
# run_config = Loihi2SimCfg(select_tag='fixed_pt', select_sub_proc_model=True)
run_config = Loihi2SimCfg()

# Run network
print("Running network")

source.run(condition=run_condition, run_cfg=run_config)

# Stop network execution
source.stop()