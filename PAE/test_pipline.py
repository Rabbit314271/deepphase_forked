
import Library.Plotting as plot
import Library.Utility as utility
from PAE import PAE

import Library.AdamWR.adamw as adamw
import Library.AdamWR.cyclic_scheduler as cyclic_scheduler


import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn
import random


import matplotlib.pyplot as plt


#Start Parameter Section
window = 1.33 #time duration of the time window
frames = 41 #sample count of the time window (60FPS)
keys = 13 #optional, used to rescale the FT window to resolution for motion controller training afterwards
joints = 24
full_range=60

input_channels = 3*joints #number of channels along time in the input data (here 3*J as XYZ-velocity component of each joint)
phase_channels = 8 #desired number of latent phase channels (usually between 2-10)

epochs = 10
batch_size = 4
learning_rate = 1e-4
weight_decay = 1e-4
restart_period = 10
restart_mult = 2
batch_size_real=2*batch_size

plotting_interval = 100 #update visualization at every n-th batch (visualization only)
pca_sequence_count = 10 #number of motion sequences visualized in the PCA (visualization only)
test_sequence_length = 80 #maximum length of each motion sequence (visualization only)
#End Parameter Section
    
def Item(value):
    return value.detach().cpu()

data_file = "../Dataset/Data.bin"
shape = utility.LoadTxtAsInt("../Dataset/Shape.txt")
sequences = utility.LoadTxtRaw("../Dataset/Sequences.txt")
#Initialize visualization
sequences = np.array(utility.Transpose2DList(sequences)[0], dtype=np.int64)
sample_count = shape[0]
feature_dim = shape[1]
test_sequences = []
for i in range(int(sequences[-1])):
    indices = np.where(sequences == (i+1))[0]
    intervals = int(np.floor(len(indices) / test_sequence_length))
    if intervals > 0:
        slices = np.array_split(indices, intervals)
        test_sequences += slices

#Initialize all seeds
seed = 23456
rng = np.random.RandomState(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
#Initialize drawing
plt.ion()
_, ax1 = plt.subplots(6,1)
_, ax2 = plt.subplots(phase_channels,5)
_, ax3 = plt.subplots(1,2)
_, ax4 = plt.subplots(2,1)
dist_amps = []
dist_freqs = []
loss_history = utility.PlottingWindow("Loss History", ax=ax4, min=0, drawInterval=plotting_interval)


#Build network model
network = utility.ToDevice(PAE(
    input_channels=input_channels,
    embedding_channels=phase_channels,
    time_range=frames,
    key_range=keys,
    window=window,
    full_range=full_range,
    enc_path='./models/test2.pth'
))
#network=torch.load('./models/complex_ep5.pth')


#Setup optimizer and loss function
optimizer = adamw.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = cyclic_scheduler.CyclicLRWithRestarts(optimizer=optimizer, batch_size=batch_size, epoch_size=sample_count, restart_period=restart_period, t_mult=restart_mult, policy="cosine", verbose=True)
loss_function = torch.nn.MSELoss()
I = np.arange(sample_count)
for epoch in range(epochs):
    scheduler.step()
    rng.shuffle(I)
    for i in range(0, sample_count, batch_size):
        print('Progress', round(100 * i / sample_count, 2), "%", end="\r")
        train_indices = I[i:i+batch_size]
        #Run model prediction
        train_batch = utility.ToDevice(torch.from_numpy(utility.ReadBatch(data_file, train_indices, feature_dim)))
        train_batch = train_batch.reshape(train_batch.shape[0], input_channels, 121)[:,:,0:120][:,:,0::2]
        #train_batch=torch.concat((train_batch[:,:,0::2],train_batch[:,:,1::2]),0)
        #print(train_batch.shape)
        

        yPred, latent, signal, params,yplot = network(train_batch)
        #train_batch = train_batch.reshape(train_batch.shape[0], -1)
        #Compute loss and train
        loss = loss_function(yPred, train_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.batch_step()
        
        #Start Visualization Section
        _a_ = Item(params[2]).squeeze().numpy()
        for i in range(_a_.shape[0]):
            dist_amps.append(_a_[i,:])
        while len(dist_amps) > 10000:
            dist_amps.pop(0)
        _f_ = Item(params[1]).squeeze().numpy()
        for i in range(_f_.shape[0]):
            dist_freqs.append(_f_[i,:])
        while len(dist_freqs) > 10000:
            dist_freqs.pop(0)
        loss_history.Add(
            (Item(loss).item(), "Reconstruction Loss")
        )
        if loss_history.Counter == 0:
            plot.Functions(ax1[0], Item(train_batch[0][:,:frames]).reshape(network.input_channels,frames), -1.0, 1.0, -5.0, 5.0, title="Motion Curves" + " " + str(network.input_channels) + "x" + str(frames), showAxes=False)
            plot.Functions(ax1[1], Item(latent[0]), -1.0, 1.0, -2.0, 2.0, title="Latent Convolutional Embedding" + " " + str(phase_channels) + "x" + str(frames), showAxes=False)
            plot.Circles(ax1[2], Item(params[0][0]).squeeze(), Item(params[2][0]).squeeze(), title="Learned Phase Timing"  + " " + str(phase_channels) + "x" + str(2), showAxes=False)
            plot.Functions(ax1[3], Item(signal[0]), -1.0, 1.0, -2.0, 2.0, title="Latent Parametrized Signal" + " " + str(phase_channels) + "x" + str(frames), showAxes=False)
            #print(Item(yplot[0]).shape,network.input_channels,frames)
            plot.Functions(ax1[4], Item(yplot[0]).reshape(network.input_channels,frames), -1.0, 1.0, -5.0, 5.0, title="Curve Reconstruction" + " " + str(network.input_channels) + "x" + str(frames), showAxes=False)

            plot.Function(ax1[5], [Item(train_batch[0]), Item(yplot[0])], -1.0, 1.0, -5.0, 5.0, colors=[(0, 0, 0), (0, 1, 1)], title="Curve Reconstruction (Flattened)" + " " + str(1) + "x" + str(network.input_channels*frames), showAxes=False)
            plot.Distribution(ax3[0], dist_amps, title="Amplitude Distribution")
            plot.Distribution(ax3[1], dist_freqs, title="Frequency Distribution")
            test_batch = utility.ToDevice(torch.from_numpy(utility.ReadBatch(data_file, random.choice(test_sequences), feature_dim)))
            test_batch = test_batch.reshape(test_batch.shape[0], input_channels, 121)[:,:,0:120]#place1
            test_batch=test_batch[:,:,0::2]

            _, _, _, params,_ = network(test_batch)
            for i in range(phase_channels):
                phase = params[0].squeeze(2)
                freq = params[1].squeeze(2)
                amps = params[2].squeeze(2)
                offs = params[3].squeeze(2)
                plot.Phase1D(ax2[i,0], Item(phase), Item(amps), color=(0, 0, 0), title=("1D Phase Values" if i==0 else None), showAxes=False)
                plot.Phase2D(ax2[i,1], Item(phase), Item(amps), title=("2D Phase Vectors" if i==0 else None), showAxes=False)
                plot.Functions(ax2[i,2], Item(freq).transpose(0,1), -1.0, 1.0, 0.0, 4.0, title=("Frequencies" if i==0 else None), showAxes=False)
                plot.Functions(ax2[i,3], Item(amps).transpose(0,1), -1.0, 1.0, 0.0, 1.0, title=("Amplitudes" if i==0 else None), showAxes=False)
                plot.Functions(ax2[i,4], Item(offs).transpose(0,1), -1.0, 1.0, -1.0, 1.0, title=("Offsets" if i==0 else None), showAxes=False)
            
            #Manifold Computation and Visualization
            pca_indices = []
            pca_batches = []
            pivot = 0
            for i in range(pca_sequence_count):
                indices = random.choice(test_sequences)
                test_batch = utility.ToDevice(torch.from_numpy(utility.ReadBatch(data_file, indices, feature_dim)))
                test_batch = test_batch.reshape(test_batch.shape[0], input_channels, 121)[:,:,0:120]#place2
                test_batch=test_batch[:,:,0::2]
                _, _, _, params,_ = network(test_batch)
                a = Item(params[2]).squeeze()
                p = Item(params[0]).squeeze()
                #Compute Phase Manifold (2D vectors composed of sin and cos terms)
                m_x = a * np.sin(2.0 * np.pi * p)
                m_y = a * np.cos(2.0 * np.pi * p)
                manifold = torch.hstack((m_x, m_y))
                pca_indices.append(pivot + np.arange(len(indices)))
                pca_batches.append(manifold)
                pivot += len(indices)
            plot.PCA2D(ax4[0], pca_indices, pca_batches, "Phase Manifold (" + str(pca_sequence_count) + " Random Sequences)")
            plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(1e-5)
        #End Visualization Section
        
    print('Epoch', epoch+1, loss_history.CumulativeValue())
