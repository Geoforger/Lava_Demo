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
# Import the data processing class and data collection class
import sys
sys.path.append("..")
from Lava_Demo.utils.utils import calculate_pooling_dim, nums_from_string
from Lava_Demo.utils.demo_loader import demoDataset
from sklearn.metrics import confusion_matrix, accuracy_score

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
                neuron_params_drop, x_size * y_size * 1, self.hidden_size, weight_norm=True), 
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


def objective(DATASET_PATH, hidden_layer):
    num_labels = 2
    num_trials = 100

    # ## Preprocess data for Network training
    # ## Variables for data input
    # Input data variables
    kernel = (4,4)
    stride = 4
    threshold = 1
    sampling_time = 1
    sample_length = 3000
    (184, 194, 120, 110)
    x_size = calculate_pooling_dim(640 - (184 + 194), kernel[1], stride, 1)
    y_size = calculate_pooling_dim(480 - (120 + 110), kernel[0], stride, 1)

    # Paths to local folders
    # Data comes from neuroTac for this script
    # Not sure if we output to anything other than the terminal at this point
    FOLDER_PATH = "/home/george/Documents/Lava_Demo/networks/"
    test_time = datetime.now()
    test_timing = test_time.strftime("%d%m%Y%H:%M:%S")
    OUTPUT_PATH = os.path.join(FOLDER_PATH, f"tests/test-{test_timing}-{sample_length}/")
    os.mkdir(OUTPUT_PATH)

    ## Set processing target
    # Check if cuda available on device for GPU processing
    if torch.cuda.is_available():
        print("Cuda available")
        print("Cuda Used")
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
    else:
        raise Exception("CUDA UNAVAILABLE")

    #############################
    # Training params
    #############################
    num_epochs = 100
    learning_rate = 0.01   # Starting learning rate
    # NOTE: Batch size is scaled down for each sample length as the samples get longer.
    batch_size = 2     #batch(sampling_length) * torch.cuda.device_count()
    num_workers = 2     # NOTE: 24 is recommended but at 6 the process dies

    factor = 3.33   # Factor by which to divide the learning rate
    lr_epoch = 10    # Lower the learning rate by factor every lr_epoch epochs
    true_rate = 0.8
    false_rate = 0.02

    # Initialise network and slayer assistant
    net = Network(hidden_size=hidden_layer, x_size=x_size, y_size=y_size)
    net.to(device)

    # TODO: 1) Play around with different optimisers
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Initialise error module
    error = slayer.loss.SpikeRate(
        true_rate=true_rate, false_rate=false_rate, reduction='sum').to(device)

    # Initialise stats and training assistants
    stats = slayer.utils.LearningStats()
    assistant = slayer.utils.Assistant(
        net, error, optimizer, stats, classifier=slayer.classifier.Rate.predict)

    ##################################
    # Initialise datasets
    ##################################
    training_set = demoDataset(DATASET_PATH, train=True, x_size=x_size, y_size=y_size, sample_length=sample_length)
    testing_set = demoDataset(DATASET_PATH, train=False, x_size=x_size, y_size=y_size, sample_length=sample_length)

    train_loader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(dataset=testing_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    ##################################
    # Training loop
    ##################################
    # Loop through each training epoch
    print("Starting training loop")

    torch.cuda.empty_cache()
    tic = time.time()
    
    testing_labels = []
    testing_preds = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}")
        epoch_tic = time.time()
        
        # Reduce learning rate by factor every lr_epoch epochs
        # Check to prevent lowering lr on first epoch
        # if epoch != 0:
        #     if(epoch % lr_epoch) == 0:
        #         learning_rate /= factor
        #         assistant.reduce_lr(factor)
        #         print(f"Learning rate reduced to {learning_rate}")

        # Training loop
        print("Training...")
        for _, (input_, label) in enumerate(train_loader):
            output = assistant.train(input_, label)
                
        # Testing loop
        print("Testing...")
        for _, (input_, label) in enumerate(test_loader):  # testing loop
            output = assistant.test(input_, label)
            if epoch == num_epochs - 1:
                for l in range(len(slayer.classifier.Rate.predict(output))):
                    testing_labels.append(label[l].cpu())
                    testing_preds.append(
                        slayer.classifier.Rate.predict(output)[l].cpu())
        
        if stats.testing.best_accuracy:
            torch.save(net.state_dict(), f"{OUTPUT_PATH}/network.pt")
        
        epoch_timing = (time.time() - epoch_tic) / 60
        print(f'\rTime taken for epoch: {np.round(epoch_timing, 2)}mins')
        
        stats.update()
        stats.save(OUTPUT_PATH + '/')
        # net.grad_flow(OUTPUT_PATH + '/')
        print(stats)
    
    # Save the best network to a hdf5 file
    net.load_state_dict(torch.load(f"{OUTPUT_PATH}/network.pt"))
    net.export_hdf5(f"{OUTPUT_PATH}/network.net")
        
    toc = time.time()
    train_timing = (toc - tic) / 60
    print("Finished training")
    print(f'\rTime taken to train network: {np.round(train_timing, 2)}mins')
    
    ###################################
    # Plot training conf matrix
    ###################################
    materials = ["0", "1"]
    materials = np.arange(num_labels)
    cnf_matrix = confusion_matrix(testing_labels, testing_preds)
    plt.figure(figsize=(10, 10))
    plt.xticks(range(len(materials)), materials)
    plt.yticks(range(len(materials)), materials)
    plt.imshow(cnf_matrix)
    plt.title('Confusion matrix')
    plt.xlabel("True Labels")
    plt.ylabel("Predicted Labels")
    plt.savefig(f"{OUTPUT_PATH}{num_epochs}training_confusion.png")
    # plt.show()
    plt.close()

    # ## Save meta data
    import csv

    # Save the hyper params to a csv file for analysis
    header = ["Artificial or Natural", "Hidden Layer Size", "Epochs", "Training Iters", "Sample Length (ms)", "True Rate", "False Rate", "Force (N)", "Accuracy (%)"]
    data = ["Natural", hidden_layer, num_epochs, num_trials, sample_length, true_rate, false_rate, 1, stats.testing.max_accuracy]

    with open(f'{OUTPUT_PATH}meta.csv', 'w', encoding='UTF8', newline='\n') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(data)
        f.close()
        
    print(output)
    output = output.cpu().numpy()
    print(np.shape(output))
    points = np.nonzero(output[0])
    plt.scatter(points[1], points[0])
    # plt.show()
    plt.close()

    # Return trainnig accuracy to optimiser
    return 1 - stats.testing.max_accuracy

# Main function of the script
def main():
    DATASET_PATH = "/home/george/Documents/Lava_Demo/data/preprocessed_data/"
    
    movements = [0, 1]
    train_test_split(DATASET_PATH, textures=movements, ratio=0.8)   # Split data based on give ratio

    for _ in range(200):
        objective(DATASET_PATH=DATASET_PATH, hidden_layer=5000)

if __name__ == "__main__":
    main()
