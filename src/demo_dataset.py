import torch
import glob
from torch.utils.data import Dataset
import sys
sys.path.append("..")
from Lava_Demo.utils.utils import nums_from_string
from Lava_Demo.utils.data_processor import DataProcessor


class DemoDataset(Dataset):
    def __init__(
        self,
        path,
        train,
        valid=False,
        x_size=640,
        y_size=480,
        sampling_time=1,
        sample_length=1000,
    ) -> None:

        super(DemoDataset, self).__init__()

        self.train = train

        # Set path to whichever dataset you want
        if self.train is True:
            self.PATH = f"{path}/train/"
        else:
            self.PATH = f"{path}/test/"

        if valid is True:
            self.PATH = f"{path}/valid/"

        self.samples = glob.glob(f"{self.PATH}*.npy")
        self.sampling_time = sampling_time
        self.num_time_bins = int(sample_length / sampling_time)
        self.x_size = x_size
        self.y_size = y_size

    # Function to retrieve spike data from index
    def __getitem__(self, index):
        filename = self.samples[index]

        # Get the folder name that contains the file for label
        label = nums_from_string(filename)[-2]

        event = DataProcessor.load_data_np(filename)
        event.create_events()
        event = event.data
        
        # print(f"Boundaries: {self.y_size} {self.x_size}")
        # print(f"Max Values: {max(event.y)} {max(event.x)}")

        spike = torch.from_numpy(
            event.to_tensor(
                sampling_time=1, dim=(1, self.y_size, self.x_size, self.num_time_bins)
            )
        ).float()

        return spike.reshape(-1, self.num_time_bins), label

    # Function to find length of dataset
    def __len__(self):
        return len(self.samples)
