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
import pickle

Restore = False
modality = "rodent"
img_size = 24000  # 2.4 second window due to downsampling by 2
n_channels = 10
sub_window_size = int(img_size / n_channels)
print("{} channels with window size {}".format(n_channels, sub_window_size))

def extract_windows_vectorized(array, sub_window_size, downsample_factor=2, overlap_factor=0.5):
    # create sliding windows of size sub_window_size, downsampling by downsample_factor, and overlapping by overlap_factor percent
    sub_windows = (
        # expand_dims are used to convert a 1D array to 2D array.
        np.expand_dims(np.arange(0, sub_window_size * downsample_factor, downsample_factor), 0) +
        np.expand_dims(np.arange(0, array.shape[-1] - sub_window_size * downsample_factor + 1,
                                 int((1-overlap_factor) * sub_window_size * downsample_factor)), 0).T)
    return array[sub_windows]

def apply_sliding_window(files, eegs):
    prep_eegs = []
    prep_files = []
    for file_name, signal in zip(files, eegs):
        # signal shape: channels x time points
        print("Input signal shape:", signal.shape)
        current_signals = []
        for channel_index in range(n_channels):
            signal_per_channel_sliding = extract_windows_vectorized(signal[channel_index], sub_window_size)
            current_signals.append(signal_per_channel_sliding)
        # batch x channels x time points
        current_signals = np.array(current_signals).transpose((1, 0, 2))
        current_file_names = np.tile([file_name], (len(current_signals),))
        print("Sliding output signal shape:", current_signals.shape)
        prep_eegs.extend(current_signals)
        prep_files.extend(current_file_names)
    prep_eegs = np.array(prep_eegs)
    prep_files = np.array(prep_files)
    return prep_eegs, prep_files

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(' Processor is %s' % (device))

# Load train EEG data with overlapping sliding windows
with open('{}_sleep_eeg_preprocessed_eegs.pickle'.format(modality), 'rb') as handle:
    train_eegs = pickle.load(handle)
with open('{}_sleep_eeg_all_eeg_files.pickle'.format(modality), 'rb') as handle:
    train_files = pickle.load(handle)
train_prep_eegs, train_prep_files = apply_sliding_window(train_files, train_eegs)

# Load test EEG data
with open('{}_eeg_preprocessed_eegs.pickle'.format(modality), 'rb') as handle:
    test_eegs = pickle.load(handle)
with open('{}_eeg_all_eeg_files.pickle'.format(modality), 'rb') as handle:
    test_files = pickle.load(handle)
test_prep_eegs, test_prep_files = apply_sliding_window(test_files, test_eegs)

# VAE model parameters for the encoder
h_layer_1 = 8
h_layer_2 = 16
h_layer_5 = 500
latent_dim = 35
kernel_size = (4, 4)
stride = 1

# VAE training parameters
batch_size = 128
epoch_num = 200
beta = 0.8

# Save folders for trained models and logs
model_save_dir = "models_{}".format(modality)
if not os.path.isdir(model_save_dir):
    os.mkdir(model_save_dir)
PATH_vae = model_save_dir + '/betaVAE_l_%2d' % (latent_dim)
results_save_dir = "results_{}".format(modality)
if not os.path.isdir(results_save_dir):
    os.mkdir(results_save_dir)

# load dataset via min-max normalization & add image channel dimension
train_imgs = np.array([(img - np.min(img)) / (np.max(img) - np.min(img))
                       for img in train_prep_eegs])[:, np.newaxis, :, :]  # batch x 1 x channels x time points
test_imgs = np.array([(img - np.min(img)) / (np.max(img) - np.min(img))
                      for img in test_prep_eegs])[:, np.newaxis, :, :]  # batch x 1 x channels x time points
print("Range of normalized images of VAE:", np.min(train_imgs), np.max(train_imgs))
train_imgs = torch.FloatTensor(train_imgs)
test_imgs = torch.FloatTensor(test_imgs)

# train with all signals
train_loader = torch.utils.data.DataLoader(train_imgs, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_imgs, batch_size=batch_size)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.econv1 = nn.Conv2d(1, h_layer_1, kernel_size=kernel_size, stride=stride)
        self.ebn1 = nn.BatchNorm2d(h_layer_1, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)
        self.econv2 = nn.Conv2d(h_layer_1, h_layer_2, kernel_size=kernel_size, stride=stride)
        self.ebn2 = nn.BatchNorm2d(h_layer_2, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)
        self.efc1  = nn.Linear(h_layer_2 * 4 * 2394, h_layer_5)
        self.edrop1 = nn.Dropout(p=0.3, inplace = False)
        self.mu_z  = nn.Linear(h_layer_5, latent_dim)
        self.logvar_z = nn.Linear(h_layer_5, latent_dim)
        #
        self.dfc1 = nn.Linear(latent_dim, h_layer_5)
        self.dfc2 = nn.Linear(h_layer_5, h_layer_2 * 4 * 2394)
        self.ddrop1 = nn.Dropout(p=0.3, inplace = False)
        self.dconv3 = nn.ConvTranspose2d(h_layer_2, h_layer_1, kernel_size=kernel_size, stride=stride, padding = 0, output_padding = 0)
        self.dbn3 = nn.BatchNorm2d(h_layer_1, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)
        self.dconv4 = nn.ConvTranspose2d(h_layer_1, 1, kernel_size=kernel_size, padding = 0, stride=stride)
        #
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def Encoder(self, x):
        eh1 = self.relu(self.ebn1(self.econv1(x)))
        eh2 = self.relu(self.ebn2(self.econv2(eh1)))
        #print(eh2.shape)
        eh5 = self.relu(self.edrop1(self.efc1(eh2.view(-1, h_layer_2 * 4 * 2394))))
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
        #print(dh2.shape)
        dh5 = self.relu(self.dbn3(self.dconv3(dh2.view(-1, h_layer_2, 4, 2394))))
        x = self.dconv4(dh5).view(-1, 1, img_size)
        #print(x.shape)
        return self.sigmoid(x)

    def forward(self, x):
        mu_z, logvar_z = self.Encoder(x)
        z = self.Reparam(mu_z, logvar_z)
        return self.Decoder(z), mu_z, logvar_z, z


# initialize model
vae = VAE()
vae.to(device)
vae_optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# loss function
SparsityLoss = nn.L1Loss(size_average = False, reduce = True)
def elbo_loss(recon_x, x, mu_z, logvar_z):

    L1loss = SparsityLoss(recon_x, x.view(-1, 1, img_size))
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
            loss_vae_value += loss_vae.data

            vae_optimizer.step()
            
        time_end = time.time()
        print('elapsed time (min) : %0.1f' % ((time_end-time_start)/60))
        print('====> Epoch: %d elbo_Loss : %0.8f' % ((i + 1), loss_vae_value / len(train_loader.dataset)))

        if i % 10 == 9:
            torch.save(vae.state_dict(), PATH_vae + "_ep_{}".format(i + 1))

if Restore:
    vae.load_state_dict(torch.load(PATH_vae, map_location=lambda storage, loc: storage))

def plot_reconstruction():
    idx = -1
    file_name_prev = test_prep_files[0]
    for file_name, img in zip(test_prep_files, test_prep_eegs):
        # counter for sliding windows over the same file
        if file_name_prev == file_name:
            idx += 1
        else:
            idx = 0
        img_variable = Variable(torch.FloatTensor(img))
        img_variable = img_variable.unsqueeze(0)
        img_variable = img_variable.to(device)
        test_imgs_z_mu, test_imgs_z_logvar = vae.Encoder(img_variable)
        test_imgs_z = vae.Reparam(test_imgs_z_mu, test_imgs_z_logvar)
        test_imgs_rec = vae.Decoder(test_imgs_z).cpu()
        # Use negative reconstruction probability as anomaly score
        test_imgs_rec = 1 - test_imgs_rec.data.numpy()
        img_i = test_imgs_rec[0].reshape(n_channels, sub_window_size)
        # Save input image and detected anomaly
        plt.figure()
        plt.imshow(img)
        plt.savefig(results_save_dir + '/{}_{}.jpg'.format(file_name, idx), bbox_inches='tight')
        plt.close()
        plt.figure()
        plt.imshow(img_i)
        plt.savefig(results_save_dir + '/Rec_l_{}_{}_{}.jpg'.format(latent_dim, file_name, idx), bbox_inches='tight')
        plt.close()
        file_name_prev = file_name

if Restore:
    plot_reconstruction()