import numpy as np
from dv_processing import noise
from datetime import timedelta
import logging
import time
import tkinter as tk
from lava.lib.dl import netx
from lava.proc import embedded_io as eio
import sys
sys.path.append("..")
from Lava_Demo.components.inivation import InivationCamera as Camera
from Lava_Demo.components.CustomInivationEncoder import (
    CustomInivationEncoder as CamEncoder,
)
from Lava_Demo.components.iniviation_visualiser import InivationVisualiser as Vis
from Lava_Demo.components.threshold_pooling import ThresholdPooling as Pooling
from Lava_Demo.components.decisions import DecisionMaker
from Lava_Demo.components.DecisionVisualiser import DecisionVisualiser as DecisionVis
from Lava_Demo.components.DecisionVisualiser import VisualiserWindow
from lava.magma.core.run_conditions import RunContinuous
from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg
# from lava.proc.io.sink import RingBuffer
from lava.magma.compiler.subcompilers.nc.ncproc_compiler import CompilerOptions

CompilerOptions.verbose = True


def main():
    loihi = False

    # Camera init
    cam_shape = (640, 480)
    filter = noise.BackgroundActivityNoiseFilter(
        cam_shape, backgroundActivityDuration=timedelta(milliseconds=10)
    )
    camera = Camera(noise_filter=filter, flatten=False, crop_params=[102, 110, 195, 170], arm_connected=False)
    print(camera.s_out.shape)

    # Init other components
    pooling = Pooling(
        in_shape=camera.out_shape, kernel=(4, 4), stride=(4, 4), threshold=1, off_events=False
    )

    # Initialise network
    net = netx.hdf5.Network(
        net_config="/home/farscope2/Documents/PhD/Lava_Demo/networks/network.net",
        sparse_fc_layer=False,
        input_shape=np.prod(
            pooling.s_out.shape,
        ),
    )
    print(net)

    # NOTE: I have an encoder that takes camera -> dense -> nx encoder
    #       This could be skipped if I could be bothered to code in C
    cam_encoder = CamEncoder(pooling.s_out.shape)
    input_vis = Vis(in_shape=pooling.out_shape, flattened_input=False)

    # Visualiser window
    root = tk.Tk()
    window = VisualiserWindow(root)
    decision_vis = DecisionVis(
        net_out_shape=net.out.shape, window=window, frequency=10
    )

    # Sim vs Loihi setup
    if loihi is True:
        in_adapter = eio.spike.PyToNxAdapter(shape=net.inp.shape)
        out_adapter = eio.spike.NxToPyAdapter(shape=net.out.shape)
        decision = DecisionMaker(
            in_shape=out_adapter.out.shape, offset=10, threshold=0.15
        )

        run_cfg = Loihi2HwCfg(select_tag="fixed_pt")
        # Connect all components
        camera.s_out.connect(pooling.a_in)
        pooling.s_out.connect(input_vis.a_in)
        pooling.s_out.connect(cam_encoder.a_in)
        cam_encoder.s_out.connect(in_adapter.inp)
        in_adapter.out.connect(net.inp)
        net.out.connect(out_adapter.inp)
        out_adapter.out.connect(decision.a_in)
    else:
        decision = DecisionMaker(
            in_shape=net.out.shape, offset=10, threshold=0.15
        )

        run_cfg = Loihi2SimCfg(select_tag="fixed_pt")
        # Connect all components
        camera.s_out.connect(pooling.a_in)
        pooling.s_out.connect(input_vis.a_in)
        pooling.s_out.connect(cam_encoder.a_in)
        cam_encoder.s_out.connect(net.inp)
        net.out.connect(decision.a_in)

    # Connect decision maker ports
    decision.s_out.connect(decision_vis.a_in)
    decision_vis.acc_in.connect_var(decision.accumulator)
    decision_vis.conf_in.connect_var(decision.confidence)

    # Set sim parameters
    run_condition = RunContinuous()

    net._log_config.level = logging.INFO
    net._log_config.level_console = logging.INFO

    print("Running Network...")
    net.run(condition=run_condition, run_cfg=run_cfg)
    print("Started sim..")
    time.sleep(1000)
    net.stop()
    print("Finished running")


if __name__ == "__main__":
    main()
