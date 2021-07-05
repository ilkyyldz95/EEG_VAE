from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data
from torch import nn, optim
import os
import time
from os import listdir
from os.path import isfile, join

modality = "rodent"
img_size = 240 * 320

def extract_windows_vectorized(array, sub_window_size, downsample_factor=2, overlap_factor=0.5):
    # create sliding windows of size sub_window_size, downsampling by downsample_factor, and overlapping by overlap_factor percent
    sub_windows = (
        # expand_dims are used to convert a 1D array to 2D array.
        np.expand_dims(np.arange(0, sub_window_size * downsample_factor, downsample_factor), 0) +
        np.expand_dims(np.arange(0, array.shape[-1] - sub_window_size * downsample_factor + 1,
                                 int((1-overlap_factor) * sub_window_size * downsample_factor)), 0).T)
    return array[sub_windows]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(' Processor is %s' % (device))

# Load train EEG data
with open('{}_sleep_eeg_preprocessed_eegs.pickle'.format(modality), 'rb') as handle:
    train_eggs = pickle.load(handle)
with open('{}_sleep_eeg_all_eeg_files.pickle'.format(modality), 'rb') as handle:
    train_files = pickle.load(handle)

# Extract overlapping sliding windows for each channel
train_prep_eegs = []
train_prep_files = []
n_channels = np.min([len(signal) for signal in train_eggs])
sub_window_size = int(img_size / n_channels)
for file_name, signal in zip(train_files, train_eggs):
    print("Input signal shape:", signal.shape)
    # signal shape: channels x time points
    current_signals = []
    for channel_index in range(n_channels):
        signal_per_channel_sliding = extract_windows_vectorized(signal[channel_index], sub_window_size)
        current_signals.append(signal_per_channel_sliding)
    # batch x channels x time points
    current_signals = np.array(current_signals).reshape((1, 0, 2))
    current_file_names = np.tile([file_name], (len(current_signals),))
    print("Sliding output signal shape:", current_signals.shape)
    train_prep_eegs.extend(current_signals)
    train_prep_files.extend(current_file_names)
# batch x channels x time points
train_prep_eegs = np.array(train_prep_eegs)

# Load test EEG data
with open('{}_eeg_preprocessed_eegs.pickle'.format(modality), 'rb') as handle:
    test_eggs = pickle.load(handle)
with open('{}_eeg_all_eeg_files.pickle'.format(modality), 'rb') as handle:
    test_files = pickle.load(handle)

# Extract overlapping sliding windows for each channel
test_prep_eegs = []
test_prep_files = []
n_channels = np.min([len(signal) for signal in test_eggs])
sub_window_size = int(img_size / n_channels)
for file_name, signal in zip(test_files, test_eggs):
    print("Input signal shape:", signal.shape)
    # signal shape: channels x time points
    current_signals = []
    for channel_index in range(n_channels):
        signal_per_channel_sliding = extract_windows_vectorized(signal[channel_index], sub_window_size)
        current_signals.append(signal_per_channel_sliding)
    # batch x channels x time points
    current_signals = np.array(current_signals).reshape((1, 0, 2))
    current_file_names = np.tile([file_name], (len(current_signals),))
    print("Sliding output signal shape:", current_signals.shape)
    test_prep_eegs.extend(current_signals)
    test_prep_files.extend(current_file_names)
# batch x channels x time points
test_prep_eegs = np.array(test_prep_eegs)

# VAE model parameters for the encoder
h_layer_1 = 32
h_layer_2 = 64
h_layer_3 = 128
h_layer_4 = 128
h_layer_5 = 2400
latent_dim = 35
kernel_size = (4, 4)
pool_size = 2
stride = 2
feature_row = 13
feature_col = 18

# VAE training parameters
batch_size = 140
epoch_num = 200
beta = 0.8

# Save folders for trained models and logs
model_save_dir = "models_{}".format(modality)
if not os.path.isdir(model_save_dir):
    os.mkdir(model_save_dir)
PATH_vae = model_save_dir + '/betaVAE_%2d' % (latent_dim)
Restore = True

# load  Dataset via min-max normalization
MIN_VAL = np.min(train_prep_eegs)
MAX_VAL = np.max(train_prep_eegs)
train_imgs = (train_prep_eegs - MIN_VAL) / (MAX_VAL - MIN_VAL)
test_imgs = (test_prep_eegs - MIN_VAL) / (MAX_VAL - MIN_VAL)
print("Range of input images of VAE:", np.min(train_imgs), np.max(train_imgs))
train_imgs = torch.FloatTensor(train_imgs)
test_imgs = torch.FloatTensor(test_imgs)

# train with all signals
train_loader = torch.utils.data.DataLoader(train_imgs, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_imgs, batch_size=batch_size)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.econv1 = nn.Conv2d(3, h_layer_1, kernel_size=kernel_size, stride=stride)
        self.ebn1 = nn.BatchNorm2d(h_layer_1, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)
        self.econv2 = nn.Conv2d(h_layer_1, h_layer_2, kernel_size=kernel_size, stride=stride)
        self.ebn2 = nn.BatchNorm2d(h_layer_2, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)
        self.econv3 = nn.Conv2d(h_layer_2, h_layer_3, kernel_size=kernel_size, stride=stride)
        self.ebn3 = nn.BatchNorm2d(h_layer_3, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)
        self.econv4 = nn.Conv2d(h_layer_3, h_layer_4, kernel_size=kernel_size, stride=stride)
        self.ebn4 = nn.BatchNorm2d(h_layer_4, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)
        self.efc1  = nn.Linear(h_layer_4 * 13 * 18, h_layer_5)
        self.edrop1 = nn.Dropout(p = 0.3, inplace = False)
        self.mu_z  = nn.Linear(h_layer_5, latent_dim)
        self.logvar_z = nn.Linear(h_layer_5, latent_dim)
        #
        self.dfc1 = nn.Linear(latent_dim, h_layer_5)
        self.dfc2 = nn.Linear(h_layer_5, h_layer_4 * 13 * 18)
        self.ddrop1 = nn.Dropout(p = 0.3, inplace = False)
        self.dconv1 = nn.ConvTranspose2d(h_layer_4, h_layer_3, kernel_size=kernel_size, stride=stride, padding = 0, output_padding = 0)
        self.dbn1 = nn.BatchNorm2d(h_layer_3, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)
        self.dconv2 = nn.ConvTranspose2d(h_layer_3, h_layer_2, kernel_size=kernel_size, stride=stride, padding = 0, output_padding = 0)
        self.dbn2 = nn.BatchNorm2d(h_layer_2, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)
        self.dconv3 = nn.ConvTranspose2d(h_layer_2, h_layer_1, kernel_size=kernel_size, stride=stride, padding = 0, output_padding = 1)
        self.dbn3 = nn.BatchNorm2d(h_layer_1, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)
        self.dconv4 = nn.ConvTranspose2d(h_layer_1, 3, kernel_size=kernel_size, padding = 0, stride=stride)

        #
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()



    def Encoder(self, x):
        eh1 = self.relu(self.ebn1(self.econv1(x)))
        eh2 = self.relu(self.ebn2(self.econv2(eh1)))
        eh3 = self.relu(self.ebn3(self.econv3(eh2)))
        eh4 = self.relu(self.ebn4(self.econv4(eh3)))
        eh5 = self.relu(self.edrop1(self.efc1(eh4.view(-1, h_layer_4 * 13 * 18))))
        mu_z = self.mu_z(eh5)
        logvar_z = self.logvar_z(eh5)
        return mu_z, logvar_z

    def Reparam(self, mu_z, logvar_z):
        std = logvar_z.mul(0.5).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        eps = eps.to(device)
        return eps.mul(std).add_(mu_z)

    def Decoder(self, z):
        dh1 = self.relu(self.dfc1(z))
        dh2 = self.relu(self.ddrop1(self.dfc2(dh1)))
        dh3 = self.relu(self.dbn1(self.dconv1(dh2.view(-1, h_layer_4, 13, 18))))
        dh4 = self.relu(self.dbn2(self.dconv2(dh3)))
        dh5 = self.relu(self.dbn3(self.dconv3(dh4)))
        x = self.dconv4(dh5).view(-1, 3, img_size)
        return self.sigmoid(x)

    def forward(self, x):
        mu_z, logvar_z = self.Encoder(x)
        z = self.Reparam(mu_z, logvar_z)
        return self.Decoder(z), mu_z, logvar_z, z

# initialize model
vae = VAE()
vae.to(device)
vae_optimizer = optim.Adam(vae.parameters(), lr = 1e-3)

# loss function
SparsityLoss = nn.L1Loss(size_average = False, reduce = True)
def elbo_loss(recon_x, x, mu_z, logvar_z):

    L1loss = SparsityLoss(recon_x, x.view(-1, 3, img_size))
    KLD = -0.5 * beta * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())

    return L1loss + KLD

# training
if Restore == False:
    print("Training...")

    for i in range(epoch_num):
        time_start = time.time()
        loss_vae_value = 0.0
        for batch_indx, data in enumerate(train_loader):
        # update VAE
            data = data
            data = Variable(data)
            data_vae = data.to(device)
            vae_optimizer.zero_grad()
            recon_x, mu_z, logvar_z, z = vae.forward(data_vae)
            loss_vae = elbo_loss(recon_x, data_vae, mu_z, logvar_z)
            loss_vae.backward()
            loss_vae_value += loss_vae.data[0]

            vae_optimizer.step()
            
        time_end = time.time()
        print('elapsed time (min) : %0.1f' % ((time_end-time_start)/60))
        print('====> Epoch: %d elbo_Loss : %0.8f' % ((i + 1), loss_vae_value / len(train_loader.dataset)))

    torch.save(vae.state_dict(), PATH_vae)

if Restore:
    vae.load_state_dict(torch.load(PATH_vae, map_location=lambda storage, loc: storage))

def plot_reconstruction():
    
    time_start = time.time()
    for indx in range(len(test_imgs)):
    # Select images

        img = test_imgs[indx]
        img_variable = Variable(torch.FloatTensor(img))
        img_variable = img_variable.unsqueeze(0)
        img_variable = img_variable.to(device)
        test_imgs_z_mu, test_imgs_z_logvar = vae.Encoder(img_variable)
        test_imgs_z = vae.Reparam(test_imgs_z_mu, test_imgs_z_logvar)
        test_imgs_rec = vae.Decoder(test_imgs_z).cpu()
        test_imgs_rec = test_imgs_rec.data.numpy()
        img_i = test_imgs_rec[0]
        img_i = img_i.transpose(1,0)
        img_i = img_i.reshape(n_channels, sub_window_size, 3)
    time_end = time.time()
    print('elapsed time (min) : %0.2f' % ((time_end-time_start)/60))
