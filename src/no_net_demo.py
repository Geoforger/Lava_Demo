from dv_processing import noise
from datetime import timedelta
import time
from lava.magma.core.run_conditions import RunContinuous
from lava.magma.core.run_configs import Loihi2SimCfg
import sys
sys.path.append("..")
from Lava_Demo.components.inivation import InivationCamera as Camera
from Lava_Demo.components.iniviation_visualiser import InivationVisualiser as Vis
from Lava_Demo.components.threshold_pooling import ThresholdPooling as Pooling


def main():
    # Camera init
    cam_shape = (640, 480)
    filter = noise.BackgroundActivityNoiseFilter(
        cam_shape, backgroundActivityDuration=timedelta(milliseconds=10)
    )
    camera = Camera(
        noise_filter=filter,
        flatten=False,
        crop_params=[102, 110, 195, 170],
        arm_connected=False,
    )
    print(camera.s_out.shape)

    # Init other components
    pooling = Pooling(
        in_shape=camera.out_shape,
        kernel=(4, 4),
        stride=(4, 4),
        threshold=1,
        off_events=True,
    )

    # NOTE: I have an encoder that takes camera -> dense -> nx encoder
    #       This could be skipped if I could be bothered to code in C
    input_vis = Vis(in_shape=pooling.out_shape, flattened_input=False)

    # Connect all components
    camera.s_out.connect(pooling.a_in)
    pooling.s_out.connect(input_vis.a_in)

    # Set sim parameters
    run_condition = RunContinuous()
    run_cfg = Loihi2SimCfg(select_tag="fixed_pt")

    print("Running Network...")
    camera.run(condition=run_condition, run_cfg=run_cfg)
    print("Started sim..")
    input("Press any key to end sim...")
    camera.stop()
    print("Finished running")


if __name__ == "__main__":
    main()
