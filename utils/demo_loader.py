import torch
import glob
from torch.utils.data import Dataset
import lava.lib.dl.slayer as slayer
from Lava_Demo.utils.utils import nums_from_string
from Lava_Demo.utils.data_processor import DataProcessor

class demoDataset(Dataset):
    def __init__(self, PATH,        # Path to folder containing the train and test subfolders
                 sampling_time=1,  # Samplet = 100us = 1ms time of neuroTac in ms, Default = 1ms
                 sample_length=2000,  # Max length of each sample in ms, Default = 5000ms
                 train=True,        # Bool for if wanting training or test data. Default = True
                 x_size=640,             # Size of the array in the x direction
                 y_size=480              # Size of the array in the y direction
    ):

        super(demoDataset, self).__init__()
        
        self.train = train
        self.PATH = PATH
        
        # Set path to whichever dataset you want
        if self.train is True:
            self.PATH = f"{PATH}/train/"
        else:
            self.PATH = f"{PATH}/test/"

        self.samples = glob.glob(f'{self.PATH}/*.npy')
        self.sampling_time = sampling_time
        self.num_time_bins = int(sample_length/sampling_time)
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
        
        spike = event.fill_tensor(                    
            torch.zeros(1, self.y_size, self.x_size,   
                        self.num_time_bins, requires_grad=False),
            sampling_time=self.sampling_time)

        return spike.reshape(-1, self.num_time_bins), label
        
    # Function to find length of dataset
    def __len__(self):
        return len(self.samples)