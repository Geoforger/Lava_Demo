# Import torch & lava libraries
import lava.lib.dl.slayer as slayer
import torch
from torch.utils.data import DataLoader
# from lava.lib.dl.netx import hdf5
import h5py
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import time
import glob
import csv

# Import the data processing class and data collection class
import sys
sys.path.append("..")
from Lava_Demo.utils.utils import calculate_pooling_dim, nums_from_string
from Lava_Demo.utils.demo_loader import demoDataset
# from sklearn.metrics import confusion_matrix, accuracy_score

# Multi GPU
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler


# ## Define network structure as before
class Network(torch.nn.Module):  # Define network
    def __init__(self, hidden_size, x_size, y_size):
        super(Network, self).__init__()

        neuron_params = {
            'threshold': 1.25,   # Previously 1.25 # THIS WAS THE CHANGE I MADE BEFORE GOING HOME 22/6/23
            'current_decay': 0.25,   # Preivously 0.25
            'voltage_decay': 0.03,  # Previously 0.03
            'tau_grad': 0.03,
            'scale_grad': 3,
            'requires_grad': True,
        }
        
        self.hidden_size = hidden_size

        neuron_params_drop = {**neuron_params,
                            'dropout': slayer.neuron.Dropout(p=0.1), }  # p=0.05

        self.blocks = torch.nn.ModuleList([
            slayer.block.cuba.Dense(
                neuron_params_drop, x_size * y_size * 1, self.hidden_size, weight_norm=True),  # , delay=True),  # 180 * 240 * 1
            slayer.block.cuba.Dense(
                neuron_params_drop, self.hidden_size, self.hidden_size, weight_norm=True),
            slayer.block.cuba.Dense(
                neuron_params_drop, self.hidden_size, 2, weight_norm=True),  # , delay=True),
        ])

        # Forward pass through for backprop
    def forward(self, spike):
        # forward computation is as simple as calling the blocks in a loop
        # Loop through each block and return the output
        for block in self.blocks:
            spike = block(spike)

        return spike

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [
            b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))


def train_test_split(PATH, textures, ratio=0.8):
    """ Function to split a given directory of data into a training and test split

    Args:
        PATH (str): Path to the data directory 
        ratio (float, optional): Ratio of training to testing data. Defaults to 0.8.
    """
    filenames = glob.glob(f"{PATH}/*.npy")
    
    if os.path.exists(f"{PATH}/train/"):
        if (input(f"Training directory exists on dataset path {PATH}. Continue? This WILL delete directory on path (y,N)") != "y"):
            raise Exception("Dataset collection exited. Dataset already exists on path")
    else:
        os.makedirs(f"{PATH}/train/", exist_ok=False)
            
    if os.path.exists(f"{PATH}/test/"):
        if (input(f"Testing directory exists on dataset path {PATH}. Continue? This WILL delete directory on path (y,N)") != "y"):
            raise Exception("Dataset collection exited. Dataset already exists on path")
    else:
        os.makedirs(f"{PATH}/test/", exist_ok=False)
            
    # Need to maintain a 80/20 split across each velocity and acceleration
    for tex in textures:
        # 1. Find all files for this given vel/acc
        l = [x for x in filenames if nums_from_string(x)[-2] == tex]
        
        # 2. Split them randomly into two lists based on ratio
        first_half = int(len(l) * ratio)
        train = l[:first_half]
        test = l[first_half:]
                    
        # 3. Copy them into the train/test folder
        for f in train:
            f_s = f.split("/")[-1]
            os.system(f"cp {f} {f'{PATH}/train/{f_s}'}")
        for f in test:
            f_s = f.split("/")[-1]
            os.system(f"cp {f} {f'{PATH}/test/{f_s}'}")

    print("Split files into train test folders")


def setup(rank, world_size):  
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def prepare(dataset, rank, world_size, batch_size=32, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader


def cleanup():
    dist.destroy_process_group()
    print("Destroyed process groups...")


def objective(rank, world_size, DATASET_PATH, OUTPUT_PATH, kernel, stride):
    # Create multi GPU server thingy
    setup(rank, world_size)
    
    num_labels = 2
    num_trials = 100
    sample_length = 2000
    x_size = calculate_pooling_dim(640 - (160 + 170), kernel[1], stride, 1)
    y_size = calculate_pooling_dim(480 - (60 + 110), kernel[0], stride, 1)

    #############################
    # Training params
    #############################
    num_epochs = 40
    learning_rate = 0.01   # Starting learning rate
    # NOTE: Batch size is scaled down for each sample length as the samples get longer.
    batch_size = 30     #batch(sampling_length) * torch.cuda.device_count()
    num_workers = 0     # NOTE: 24 is recommended but at 6 the process dies

    hidden_layer = 1250
    factor = 3.33   # Factor by which to divide the learning rate
    lr_epoch = 20    # Lower the learning rate by factor every lr_epoch epochs
    true_rate = 0.8
    false_rate = 0.02

    # Create network and distributed object
    net = Network(hidden_size=hidden_layer, x_size=x_size, y_size=y_size).to(rank)
    net = DDP(net, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # TODO: 1) Play around with different optimisers
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Initialise error module
    error = slayer.loss.SpikeRate(
        true_rate=true_rate, false_rate=false_rate, reduction='sum').to(rank)

    # Initialise stats and training assistants
    stats = slayer.utils.LearningStats()
    assistant = slayer.utils.Assistant(
        net, error, optimizer, stats, classifier=slayer.classifier.Rate.predict)

    ##################################
    # Initialise datasets
    ##################################
    training_set = demoDataset(DATASET_PATH, train=True, x_size=x_size, y_size=y_size, sample_length=sample_length)
    testing_set = demoDataset(DATASET_PATH, train=False, x_size=x_size, y_size=y_size, sample_length=sample_length)

    train_loader = prepare(training_set, rank, world_size, batch_size=batch_size, num_workers=num_workers)
    test_loader = prepare(testing_set, rank, world_size, batch_size=batch_size, num_workers=num_workers)

    ##################################
    # Training loop
    ##################################
    # Loop through each training epoch
    print("Starting training loop")

    # torch.cuda.empty_cache()
    tic = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}")
        epoch_tic = time.time()
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)
        
        # Reduce learning rate by factor every lr_epoch epochs
        # Check to prevent lowering lr on first epoch
        if epoch != 0:
            if(epoch % lr_epoch) == 0:
                learning_rate /= factor
                assistant.reduce_lr(factor)
                print(f"Learning rate reduced to {learning_rate}")

        # Training loop
        print("Training...")
        for _, (input_, label) in enumerate(train_loader):
            output = assistant.train(input_, label)
                
        # Testing loop
        print("Testing...")
        for _, (input_, label) in enumerate(test_loader): 
            output = assistant.test(input_, label)
        
        # TODO: This should be changed to look at the output stats file instead of the stats object
        if stats.testing.best_accuracy:
            torch.save(net.module.state_dict(), f"{OUTPUT_PATH}/network.pt")
        
        epoch_timing = (time.time() - epoch_tic) / 60
        print(f'\rTime taken for epoch: {np.round(epoch_timing, 2)}mins')
        
        stats.update()
        stats.save(OUTPUT_PATH + '/')
        # net.grad_flow(OUTPUT_PATH + '/')
        print(stats)
    
    # TODO: This should be changed to look at the output stats file instead of the stats object
    # Save the best network to a hdf5 file
    time.sleep(2)
    net.module.load_state_dict(torch.load(f"{OUTPUT_PATH}/network.pt"))
    net.module.export_hdf5(f"{OUTPUT_PATH}/network.net")
        
    toc = time.time()
    train_timing = (toc - tic) / 60
    print("Finished training")
    print(f'\rTime taken to train network: {np.round(train_timing, 2)}mins')

    # # ## Save meta data
    header = ["Hidden Layer Size", "Epochs", "Training Iters", "Sample Length (ms)", "True Rate", "False Rate", "Accuracy (%)", "X Size", "Y Size", "Pooling Kernel", "Stride"]
    data = [hidden_layer, num_epochs, num_trials, sample_length, true_rate, false_rate, stats.testing.max_accuracy, x_size, y_size, kernel, stride]

    with open(f'{OUTPUT_PATH}/meta.csv', 'w', encoding='UTF8', newline='\n') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(data)
        f.close()

    cleanup()
    time.sleep(3)
    
    # Return training accuracy to optimiser
    return 1 - stats.testing.max_accuracy


# Main function of the script
def main():
    DATASET_PATH = "/home/george/Documents/Lava_Demo/data/preprocessed_data/"
    movements = [0, 1]

    kernel = (4,4)
    stride = 4
    train_test_split(DATASET_PATH, textures=movements, ratio=0.8)   # Split data based on give ratio
    
    for _ in range(100):
        # # Paths to local folders
        # Data comes from neuroTac for this script
        # Not sure if we output to anything other than the terminal at this point
        FOLDER_PATH = "/home/george/Documents/Lava_Demo/networks/"
        test_time = datetime.now()
        test_timing = test_time.strftime("%d%m%Y%H:%M:%S")
        OUTPUT_PATH = os.path.join(FOLDER_PATH, f"tests/multi_gpu-{test_timing}/")
        os.mkdir(OUTPUT_PATH)
        
        world_size = 3
        mp.spawn(
            objective,
            args=[world_size, DATASET_PATH, OUTPUT_PATH, kernel, stride],
            nprocs=world_size
        )


if __name__ == "__main__":
    main()