# Import torch & lava libraries
import lava.lib.dl.slayer as slayer
import torch
from torch.utils.data import DataLoader

# from lava.lib.dl.netx import hdf5
import h5py
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from ast import literal_eval
import numpy as np
from sklearn.metrics import accuracy_score

# Import the data processing class and data collection class
import sys
sys.path.append("..")
from Lava_Demo.src.demo_dataset import DemoDataset
from Lava_Demo.utils.utils import dataset_split

# Multi GPU
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler


# Define network structure
class Network(torch.nn.Module):
    def __init__(self, output_neurons, x_size, y_size, dropout=0.1):
        super(Network, self).__init__()

        neuron_params = {
            "threshold": 1.15,  # Previously 1.25
            "current_decay": 0.25,  # Preivously 0.25
            "voltage_decay": 0.03,  # Previously 0.03
            "tau_grad": 0.03,
            "scale_grad": 3,
            "requires_grad": True,
        }

        self.output_neurons = int(output_neurons)
        self.hidden_layer = 350

        neuron_params_drop = {
            **neuron_params,
            "dropout": slayer.neuron.Dropout(p=dropout),
        }

        self.blocks = torch.nn.ModuleList(
            [
                slayer.block.cuba.Dense(
                    neuron_params_drop,
                    x_size * y_size * 1,
                    self.hidden_layer,
                    weight_norm=True,
                ),
                slayer.block.cuba.Dense(
                    neuron_params_drop,
                    self.hidden_layer,
                    self.output_neurons,
                    weight_norm=True,
                ),
            ]
        )

    def forward(self, spike):
        for block in self.blocks:
            spike = block(spike)

        return spike

    def grad_flow(self, path):
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, "synapse")]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + "gradFlow.png")
        plt.close()

        return grad

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))


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


def objective(rank, world_size, OUTPUT_PATH, DATASET_PATH, true_rate):
    setup(rank, world_size)

    #############################
    # Output params
    #############################
    OUTPUT_PATH = f"{OUTPUT_PATH}{int(time.time())}/"
    if rank == 0:
        os.makedirs(OUTPUT_PATH, exist_ok=False)

    # Read meta for x, y sizes of preproc data
    meta = pd.read_csv(f"{DATASET_PATH}/meta.csv")
    meta['output_shape'] = meta['output_shape'].apply(literal_eval)
    x_size, y_size = meta["output_shape"].iloc[0]

    #############################
    # Training params
    #############################
    num_epochs = 100
    learning_rate = 0.001   # Starting learning rate
    batch_size = 1
    hidden_layer = 125
    factor = 3.33   # Factor by which to divide the learning rate
    lr_epoch = 200    # Lower the learning rate by factor every lr_epoch epochs
    sample_length = 1000
    # truer_rate = 0.5

    # Initialise network and slayer assistant
    net = Network(
        output_neurons=3,
        x_size=x_size,
        y_size=y_size,
        dropout=0.3
    ).to(rank)
    if rank == 0:
        print(net)

    # Create network and distributed object
    net = DDP(net, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # TODO: 1) Play around with different optimisers
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Initialise error module
    # TODO: 1) Play around with different error rates, etc.
    error = slayer.loss.SpikeRate(
        true_rate=true_rate, false_rate=0.02, reduction='sum').to(rank)

    # Initialise stats and training assistants
    stats = slayer.utils.LearningStats()
    assistant = slayer.utils.Assistant(
        net, error, optimizer, stats, classifier=slayer.classifier.Rate.predict)

    # Load in datasets
    training_set = DemoDataset(
        DATASET_PATH,
        train=True,
        x_size=x_size,
        y_size=y_size,
    )
    testing_set = DemoDataset(
        DATASET_PATH,
        train=False,
        x_size=x_size,
        y_size=y_size,
    )

    train_loader = prepare(training_set, rank, world_size, batch_size=batch_size, num_workers=0)
    test_loader = prepare(testing_set, rank, world_size, batch_size=batch_size, num_workers=0)

    testing_labels = []
    testing_preds = []
    speed_labels = []
    depth_labels = []

    ##################################
    # Training loop
    ##################################
    # Loop through each training epoch
    if rank == 0:
        print("Starting training loop")
    for epoch in range(num_epochs):
        tic = time.time()
        if rank == 0:
            print(f"\nEpoch {epoch}")
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)
        dist.barrier()

        # Reduce learning rate by factor every lr_epoch epochs
        # Check to prevent lowering lr on first epoch
        if epoch != 0:
            if (epoch % lr_epoch) == 0:
                learning_rate /= factor
                assistant.reduce_lr(factor)
                print(f"Learning rate reduced to {learning_rate}")

        # Training loop
        if rank == 0:
            print("Training...")
        for _, (input, label) in enumerate(train_loader):
            output = assistant.train(input, label)

        # Testing loop
        if rank == 0:
            print("Testing...")
        for _, (input, label) in enumerate(test_loader):
            output = assistant.test(input, label)
            if epoch == num_epochs - 1:
                for o in range(len(slayer.classifier.Rate.predict(output))):
                    testing_labels.append(int(label[o].cpu()))
                    testing_preds.append(
                        int(slayer.classifier.Rate.predict(output)[o].cpu())
                    )

        dist.barrier()  # Wait for all ranks to finish epoch

        if stats.testing.best_accuracy:
            if rank == 0:
                torch.save(net.module.state_dict(), f"{OUTPUT_PATH}/network.pt")

        stats.update()
        stats.save(OUTPUT_PATH + "/")

        toc = time.time()
        epoch_timing = (toc - tic) / 60
        if rank == 0:
            print(stats)
            print(f"\r[Epoch {epoch:2d}/{num_epochs}] {stats}", end="")
            print(f"\r[Epoch {epoch:2d}/{num_epochs}] {stats}", end="")
            print(f"\nTime taken for this epoch = {epoch_timing} mins")

    # TODO: This should be changed to look at the output stats file instead of the stats object
    # Save the best network to a hdf5 file
    if rank == 0:
        print("Finished training")
        net.module.load_state_dict(torch.load(f"{OUTPUT_PATH}/network.pt"))
        net.module.export_hdf5(f"{OUTPUT_PATH}/network.net")
        print(f"Validation accuracy: {accuracy_score(testing_labels, testing_preds)}")

        # Save output stats for testing
        test_stats = pd.DataFrame(data={
            "Labels": testing_labels,
            "Predictions": testing_preds,
            "True Rate": true_rate
        })
        test_stats.to_csv(f"{OUTPUT_PATH}/output_labels.csv")
        print(f"Accuracy of test: {stats.testing.best_accuracy}")

    cleanup()


def main():
    DATASET_PATH = (
        "/media/george/My Passport/George/George Datasets/lava_demo_preprocessed/"
    )
    OUTPUT_PATH = "/home/george/Documents/Lava_Demo/networks/net_test_"
    train_ratio = 0.8

    # Train test split
    print(f"Splitting dataset into train/test with ratio: {train_ratio}")
    dataset_split(DATASET_PATH, train_ratio=train_ratio, valid_ratio=None)

    world_size = 3

    for r in np.arange(0.7, 1.0, 0.1):
        for _ in range(25):
            mp.spawn(
                objective,
                args=[world_size, OUTPUT_PATH, DATASET_PATH, r],
                nprocs=world_size
            )
            time.sleep(3)


if __name__ == "__main__":
    main()
