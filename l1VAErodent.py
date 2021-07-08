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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

Restore = True
modality = "rodent"
fs = 2000
img_size = 24000  # 2.4 second window due to downsampling by 2
n_channels = 10
sub_window_size = int(img_size / n_channels)
downsample_factor = 2
print("{} channels with window size {}".format(n_channels, sub_window_size))

def extract_windows_vectorized(array, sub_window_size, downsample_factor, overlap_factor=0.5):
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
            signal_per_channel_sliding = extract_windows_vectorized(signal[channel_index], sub_window_size, downsample_factor)
            current_signals.append(signal_per_channel_sliding)
        # batch x channels x time points
        current_signals = np.array(current_signals).transpose((1, 0, 2))
        print("Sliding output signal shape:", current_signals.shape)
        # take file name only
        file_name = file_name.split("/")[-1].split(".")[0]
        current_file_names = np.tile([file_name], (len(current_signals),))
        prep_eegs.extend(current_signals)
        prep_files.extend(current_file_names)
    prep_eegs = np.array(prep_eegs)
    print("Dataset shape:", prep_eegs.shape)
    prep_files = np.array(prep_files)
    return prep_eegs, prep_files

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(' Processor is %s' % (device))

# VAE model parameters for the encoder
h_layer_1 = 8
h_layer_2 = 16
h_layer_5 = 1000
latent_dim = 64
kernel_size = (4, 4)
stride = 1

# VAE training parameters
batch_size = 128
epoch_num = 350
beta = 0.8

# Save folders for trained models and logs
model_save_dir = "models_{}".format(modality)
if not os.path.isdir(model_save_dir):
    os.mkdir(model_save_dir)
PATH_vae = model_save_dir + '/betaVAE_l_%2d' % (latent_dim)
results_save_dir = "results_{}".format(modality)
if not os.path.isdir(results_save_dir):
    os.mkdir(results_save_dir)

# Load train EEG data with overlapping sliding windows
with open('{}_sleep_eeg_preprocessed_eegs.pickle'.format(modality), 'rb') as handle:
    train_eegs = pickle.load(handle)
with open('{}_sleep_eeg_all_eeg_files.pickle'.format(modality), 'rb') as handle:
    train_files = pickle.load(handle)
train_prep_eegs, train_files = apply_sliding_window(train_files, train_eegs)
# load dataset via min-max normalization & add image channel dimension
train_imgs = np.array([(img - np.min(img)) / (np.max(img) - np.min(img))
                       for img in train_prep_eegs])[:, np.newaxis, :, :]  # batch x 1 x channels x time points
print("Range of normalized images of VAE:", np.min(train_imgs), np.max(train_imgs))
train_imgs = torch.FloatTensor(train_imgs)
train_loader = torch.utils.data.DataLoader(train_imgs, batch_size=batch_size, shuffle=True)

if Restore:
    # Load test EEG data
    with open('{}_eeg_preprocessed_eegs.pickle'.format(modality), 'rb') as handle:
        test_eegs = pickle.load(handle)
    with open('{}_eeg_all_eeg_files.pickle'.format(modality), 'rb') as handle:
        test_files = pickle.load(handle)
    test_prep_eegs, test_files = apply_sliding_window(test_files, test_eegs)
    test_imgs = np.array([(img - np.min(img)) / (np.max(img) - np.min(img))
                          for img in test_prep_eegs])[:, np.newaxis, :, :]  # batch x 1 x channels x time points
    test_imgs = torch.FloatTensor(test_imgs)
    # Test on sleep as well
    test_files = np.concatenate([train_files, test_files], 0)
    test_imgs = np.concatenate([train_imgs, test_imgs], 0)

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
        self.scale_z = nn.Linear(h_layer_5, latent_dim)
        #
        self.dfc1 = nn.Linear(latent_dim, h_layer_5)
        self.dfc2 = nn.Linear(h_layer_5, h_layer_2 * 4 * 2394)
        self.ddrop1 = nn.Dropout(p=0.3, inplace = False)
        self.dconv3 = nn.ConvTranspose2d(h_layer_2, h_layer_1, kernel_size=kernel_size, stride=stride, padding = 0, output_padding = 0)
        self.dbn3 = nn.BatchNorm2d(h_layer_1, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)
        self.dconv4 = nn.ConvTranspose2d(h_layer_1, 1, kernel_size=kernel_size, padding = 0, stride=stride)
        #
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()
        self.softplus = nn.Softplus()
        # initialize weights
        torch.nn.init.xavier_uniform_(self.econv1.weight)
        self.econv1.bias.data.fill_(0.01)
        torch.nn.init.xavier_uniform_(self.econv2.weight)
        self.econv2.bias.data.fill_(0.01)
        torch.nn.init.xavier_uniform_(self.efc1.weight)
        self.efc1.bias.data.fill_(0.01)
        torch.nn.init.xavier_uniform_(self.mu_z.weight)
        self.mu_z.bias.data.fill_(0.01)
        torch.nn.init.xavier_uniform_(self.scale_z.weight)
        self.scale_z.bias.data.fill_(0.01)
        torch.nn.init.xavier_uniform_(self.dfc1.weight)
        self.dfc1.bias.data.fill_(0.01)
        torch.nn.init.xavier_uniform_(self.dfc2.weight)
        self.dfc2.bias.data.fill_(0.01)
        torch.nn.init.xavier_uniform_(self.dconv3.weight)
        self.dconv3.bias.data.fill_(0.01)
        torch.nn.init.xavier_uniform_(self.dconv4.weight)
        self.dconv4.bias.data.fill_(0.01)

    def Encoder(self, x):
        eh1 = self.relu(self.ebn1(self.econv1(x)))
        eh2 = self.relu(self.ebn2(self.econv2(eh1)))
        #print(eh2.shape)
        eh5 = self.relu(self.edrop1(self.efc1(eh2.view(-1, h_layer_2 * 4 * 2394))))
        mu_z = self.mu_z(eh5)
        scale_z = self.scale_z(eh5)
        return mu_z, scale_z

    def Reparam(self, mu_z, scale_z):
        #std = logvar_z.mul(0.5).exp()
        #eps = Variable(std.data.new(std.size()).normal_())
        scale = self.softplus(scale_z)
        eps = torch.randn(mu_z.shape)
        eps = eps.to(device)
        return mu_z + scale * eps  # eps.mul(std).add_(mu_z)

    def Decoder(self, z):
        dh1 = self.relu(self.dfc1(z))
        dh2 = self.relu(self.ddrop1(self.dfc2(dh1)))
        #print(dh2.shape)
        dh5 = self.relu(self.dbn3(self.dconv3(dh2.view(-1, h_layer_2, 4, 2394))))
        x = self.dconv4(dh5).view(-1, 1, img_size)
        #print(x.shape)
        return self.sigmoid(x)

    def forward(self, x):
        mu_z, scale_z = self.Encoder(x)
        z = self.Reparam(mu_z, scale_z)
        return self.Decoder(z), mu_z, scale_z, z


# initialize model
vae = VAE()
vae.to(device)
print(vae)
vae_optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# loss function
SparsityLoss = nn.L1Loss(size_average = False, reduce = True)
def elbo_loss(recon_x, x, mu_z, scale_z):

    L1loss = SparsityLoss(recon_x, x.view(-1, 1, img_size))
    KLD = -0.5 * beta * torch.sum(1 + torch.log(scale_z.pow(2) + 1e-6) - mu_z.pow(2) - scale_z.pow(2))

    return torch.mean(L1loss + KLD, dim=0)

# training
if not Restore:
    print("Training...")
    best_loss = 1e10
    for i in range(epoch_num):
        time_start = time.time()
        loss_vae_value = 0.0
        for batch_indx, data in enumerate(train_loader):
            data = Variable(data)
            data_vae = data.to(device)

            vae_optimizer.zero_grad()
            recon_x, mu_z, scale_z, z = vae.forward(data_vae)
            loss_vae = elbo_loss(recon_x, data_vae, mu_z, scale_z)

            loss_vae.backward()
            #torch.nn.utils.clip_grad_norm_(vae.parameters(), 1)
            loss_vae_value += loss_vae.data
            vae_optimizer.step()
            
        time_end = time.time()
        print('elapsed time (min) : %0.1f' % ((time_end-time_start)/60))
        current_loss = loss_vae_value / len(train_loader.dataset)
        print('====> Epoch: %d elbo_Loss : %0.8f' % ((i + 1), current_loss))
        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = i + 1
            torch.save(vae.state_dict(), PATH_vae + "_best")
        if i % 10 == 9:
            torch.save(vae.state_dict(), PATH_vae + "_ep_{}".format(i + 1))
    print("Best loss and epoch:", best_loss, best_epoch)

if Restore:
    vae.load_state_dict(torch.load(PATH_vae + "_best", map_location=lambda storage, loc: storage))

def plot_reconstruction():
    vae.eval()
    idx = -1
    file_name_prev = test_files[0]
    latent_vars = []
    for file_name, img in zip(test_files, test_imgs):
        # counter for sliding windows over the same file
        if file_name_prev == file_name:
            idx += 1
        else:
            idx = 0
        img_variable = Variable(torch.FloatTensor(img))
        img_variable = img_variable.unsqueeze(0)
        img_variable = img_variable.to(device)
        test_imgs_z_mu, test_imgs_z_scale = vae.Encoder(img_variable)
        test_imgs_rec = []
        # Repeat and average reconstruction
        for _ in range(10):
            test_imgs_z = vae.Reparam(test_imgs_z_mu, test_imgs_z_scale)
            test_imgs_rec.append(vae.Decoder(test_imgs_z).squeeze(0).squeeze(0).cpu().data.numpy())  # img_size vector
        # Use reconstruction error as anomaly score
        latent_vars.append(test_imgs_z.squeeze(0).squeeze(0).cpu().data.numpy())
        img_i = np.mean(test_imgs_rec, 0).reshape(n_channels, sub_window_size)
        img = img.reshape(n_channels, sub_window_size)
        img_anom = np.abs(img - img_i)
        # Save input image, its spectogram, and detected anomaly
        T = np.arange(1, sub_window_size + 1) / (fs/downsample_factor)
        plt.figure()
        _, axs = plt.subplots(int(n_channels / 2), 2)
        for ch in range(int(n_channels / 2)):
            axs[ch, 0].plot(T, img[ch], "b")
            axs[ch, 0].plot(T, img_anom[ch], "r")
            axs[ch, 1].specgram(img[ch], Fs=fs/downsample_factor)
            axs[ch, 0].grid()
            axs[ch, 1].grid()
            axs[ch, 0].set_ylim([0, 1])
            axs[ch, 1].set_ylim([0, 60])
        axs[-1, 0].set(xlabel="Time (s) vs. Input (b) Anomaly (r)")
        axs[-1, 1].set(xlabel="Time (s) vs. Frequency (Hz)")
        plt.savefig(results_save_dir + '/{}_{}_ch1_l_{}.jpg'.format(file_name, idx, latent_dim), bbox_inches='tight')
        plt.close()
        plt.figure()
        _, axs = plt.subplots(int(n_channels / 2), 2)
        for ch in range(int(n_channels / 2)):
            axs[ch, 0].plot(T, img[ch + int(n_channels / 2)], "b")
            axs[ch, 0].plot(T, img_anom[ch + int(n_channels / 2)], "r")
            axs[ch, 1].specgram(img[ch + int(n_channels / 2)], Fs=fs/downsample_factor)
            axs[ch, 0].grid()
            axs[ch, 1].grid()
            axs[ch, 0].set_ylim([0, 1])
            axs[ch, 1].set_ylim([0, 60])
        axs[-1, 0].set(xlabel="Time (s) vs. Input (b) Anomaly (r)")
        axs[-1, 1].set(xlabel="Time (s) vs. Frequency (Hz)")
        plt.savefig(results_save_dir + '/{}_{}_ch2_l_{}.jpg'.format(file_name, idx, latent_dim), bbox_inches='tight')
        plt.close()

        file_name_prev = file_name
    # Plot latent space w.r.t. categories
    idx_awake = [idx for idx in range(len(test_files)) if "awake" in test_files[idx] or "Awake" in test_files[idx]]
    idx_sleep = [idx for idx in range(len(test_files)) if "sleep" in test_files[idx] or "Sleep" in test_files[idx]]
    idx_light = [idx for idx in range(len(test_files)) if "light" in test_files[idx] or "Light" in test_files[idx]]
    idx_dark = [idx for idx in range(len(test_files)) if "dark" in test_files[idx] or "Dark" in test_files[idx]]
    print("Latent space matrix original shape:", np.array(latent_vars).shape)
    latent_vars_embedded = TSNE(n_components=2).fit_transform(np.array(latent_vars))
    print("Latent space matrix reduced shape:", latent_vars_embedded.shape)
    plt.figure()
    _, ax = plt.subplots()
    ax.scatter(latent_vars_embedded[idx_awake, 0], latent_vars_embedded[idx_awake, 1], c="b", label="Awake")
    ax.scatter(latent_vars_embedded[idx_sleep, 0], latent_vars_embedded[idx_sleep, 1], c="r", label="Sleep")
    ax.legend()
    plt.savefig(results_save_dir + '/latent_awake_sleep_l_{}.jpg'.format(latent_dim), bbox_inches='tight')
    plt.close()
    plt.figure()
    _, ax = plt.subplots()
    ax.scatter(latent_vars_embedded[idx_light, 0], latent_vars_embedded[idx_light, 1], c="b", label="Light")
    ax.scatter(latent_vars_embedded[idx_dark, 0], latent_vars_embedded[idx_dark, 1], c="r", label="Dark")
    ax.legend()
    plt.savefig(results_save_dir + '/latent_dark_light_l_{}.jpg'.format(latent_dim), bbox_inches='tight')
    plt.close()

if Restore:
    plot_reconstruction()