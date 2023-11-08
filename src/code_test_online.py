import sys
sys.path.append("..")
sys.path.append(".")

import unittest
import numpy as np
import typing as ty
import time

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from dv_stream import DvStream, DvStreamPM

from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.process.ports.ports import OutPort
from lava.utils.events import sub_sample, encode_data_and_indices

from components.online_visualiser import OnlineVisualiser
import matplotlib.pyplot as plt

class RecvSparse(AbstractProcess):
    """Process that receives arbitrary sparse data.

    Parameters
    ----------
    shape: tuple
        Shape of the InPort and Vars.
    """

    def __init__(self,
                 shape: ty.Tuple[int]) -> None:
        super().__init__(shape=shape)

        self.in_port = InPort(shape=shape)

        self.data = Var(shape=shape, init=np.zeros(shape, dtype=int))
        self.idx = Var(shape=shape, init=np.zeros(shape, dtype=int))


@implements(proc=RecvSparse, protocol=LoihiProtocol)
@requires(CPU)
class PyRecvSparsePM(PyLoihiProcessModel):
    """Receives sparse data from PyInPort and stores a padded version of
    received data and indices in Vars."""
    in_port: PyInPort = LavaPyType(PyInPort.VEC_SPARSE, int)

    data: np.ndarray = LavaPyType(np.ndarray, int)
    idx: np.ndarray = LavaPyType(np.ndarray, int)

    def run_spk(self) -> None:
        data, idx = self.in_port.recv()
        print(data)

        self.data = np.pad(data,
                           pad_width=(0, self.in_port.shape[0] - data.shape[0]))
        self.idx = np.pad(idx,
                          pad_width=(0, self.in_port.shape[0] - data.shape[0]))


def main():
    

    max_num_events = 15
    shape_frame_in = (240, 180)
    dv_stream = DvStream(address="127.0.0.1",
                            port=52559,
                            shape_out=(max_num_events,),
                            shape_frame_in=shape_frame_in)

    recv_sparse = RecvSparse(shape=(max_num_events,))
    vis = OnlineVisualiser(in_shape=(max_num_events,), sample_length=max_num_events, frame_shape=shape_frame_in)

    # dv_stream.out_port.connect(recv_sparse.in_port)
    dv_stream.out_port.connect(vis.a_in)

    num_steps = 100
    run_cfg = Loihi1SimCfg()
    run_cnd = RunSteps(num_steps=100)

    # F = lambda x: np.sin(2*x)

    # plt.ion()    
    # x = np.linspace(0, 1, 200)
    # plt.plot(x, F(x))


    # for i in range(100):
    #     if 'ax' in globals(): ax.remove()
    #     newx = np.random.choice(x, size = 10)
    #     ax = plt.scatter(newx, F(newx))
    #     plt.pause(0.05)

    # plt.ioff()
    # plt.show()
    dv_stream.run(condition=run_cnd, run_cfg=run_cfg)
    # for i in range(num_steps):
    #     dv_stream.run(condition=run_cnd, run_cfg=run_cfg)

        # received_data = recv_sparse.data.get()
        # received_indices = recv_sparse.idx.get()


    dv_stream.stop()

if __name__ == '__main__':
        __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
        main()