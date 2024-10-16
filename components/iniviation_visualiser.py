import numpy as np
import cv2 as cv

from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyVarPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import InPort, VarPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol


class InivationVisualiser(AbstractProcess):
    """
    Process that visualises events from an Inivation camera process.

    Parameters:
    -----------
        in_shape (tuple):
            Shape of the incoming camera connection
        flatten (bool, optional):
            Whether to flatten the output data array. Defaults to False.
        window_name (string, optional):
            Name of the CV window. Defaults 'Spike Visualiser'
    Returns:
        None
    """
    def __init__(
        self,
        in_shape,
        flattened_input=False,
        window_name="Spike Visualiser"
    ) -> None:

        self.flattened_input = flattened_input
        self.window_name = window_name

        if self.flattened_input:
            self.in_shape = (np.prod(in_shape),)
        else:
            self.in_shape = in_shape

        self.a_in = InPort(shape=self.in_shape)

        super().__init__(
            in_shape=self.in_shape,
            flattened_input=self.flattened_input,
            window_name=self.window_name
        )


@implements(proc=InivationVisualiser, protocol=LoihiProtocol)
@requires(CPU)
class PySparseInivationVisualiserModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_SPARSE, np.float64)

    def __init__(self, proc_params) -> None:
        super().__init__(proc_params)
        self.in_shape = proc_params["in_shape"]
        self.flattened_input = proc_params["flattened_input"]
        self.window_name = proc_params["window_name"]

        self.window = cv.namedWindow(f"{self.window_name}", cv.WINDOW_NORMAL)

    def preview_events(self, colour_image) -> None:
        cv.imshow(f"{self.window_name}", colour_image)
        cv.waitKey(2)

    def run_spk(self) -> None:
        data, indices = self.a_in.recv()
        colour_image = np.zeros(
            (self.in_shape[0], self.in_shape[1], 3), dtype=np.uint8)

        on_events = indices[np.where(data == 1)]
        off_events = indices[np.where(data == 0)]

        on_x, on_y = np.unravel_index(on_events, self.in_shape)
        off_x, off_y = np.unravel_index(off_events, self.in_shape)

        # Set all on events to green and off events to red
        colour_image[on_x, on_y] = [0, 0, 255]
        colour_image[off_x, off_y] = [0, 255, 0]

        # Make image larger in frame
        resized_image = cv.resize(colour_image, (800, 600))

        # self.preview_events(colour_image)
        self.preview_events(resized_image)

    def _pause(self) -> None:
        """Pause was called by the runtime"""
        super()._pause()

    def _stop(self) -> None:
        """Stop was called by the runtime"""
        cv.destroyAllWindows()
        super()._stop()
