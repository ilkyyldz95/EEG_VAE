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
from scipy.signal import spectrogram
from scipy.stats import ttest_ind
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, classification_report

Restore = True
modality = "human"
fs = 250.0
img_size = 14400
n_channels = 48
sub_window_size = int(img_size / n_channels)  # sub_window_size / (fs / downsample_factor) second window
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
        print("Input signal shape channels x time points:", signal.shape)
        current_signals = []
        for channel_index in range(signal.shape[0]):
            signal_per_channel_sliding = extract_windows_vectorized(signal[channel_index], sub_window_size, downsample_factor)
            current_signals.append(signal_per_channel_sliding)
        # replicate through channels if there are less channels than max n_channels
        current_signals = np.array(current_signals)
        print("Sliding signal shape channels x batch x time points:", current_signals.shape)
        if signal.shape[0] < n_channels:
            current_signals = np.tile(current_signals,
                                      [int(np.ceil(n_channels/signal.shape[0])), 1, 1])
            current_signals = current_signals[:n_channels]
        # batch x channels x time points
        current_signals = current_signals.transpose((1, 0, 2))
        print("Sliding output signal shape batch x channels x time points:", current_signals.shape)
        # take file name only
        file_name = file_name.split("/")[-1].split(".edf")[0]
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
checkpoint_epoch = 0
epoch_num = 250
beta = 0.8

# Save folders for trained models and logs
model_save_dir = "models_{}".format(modality)
if not os.path.isdir(model_save_dir):
    os.mkdir(model_save_dir)
PATH_vae = model_save_dir + '/betaVAE_l_{}_input_{}'.format(latent_dim, img_size)
results_save_dir = "results_{}".format(modality)
if not os.path.isdir(results_save_dir):
    os.mkdir(results_save_dir)

# Load train EEG data with overlapping sliding windows
with open('{}_train_eegs.pickle'.format(modality), 'rb') as handle:
    train_eegs = pickle.load(handle)
with open('{}_train_files.pickle'.format(modality), 'rb') as handle:
    train_files = pickle.load(handle)
train_prep_eegs, train_files = apply_sliding_window(train_files, train_eegs)
# load dataset via min-max normalization & add image channel dimension
train_imgs = np.array([(img - np.min(img)) / (np.max(img) - np.min(img))
                       for img in train_prep_eegs])[:, np.newaxis, :, :]  # batch x 1 x channels x time points
print("Range of normalized images of VAE:", np.min(train_imgs), np.max(train_imgs))

# separate normal signals into train and test portions
shuffled_idx = np.random.permutation(range(len(train_imgs)))
train_idx = shuffled_idx[:int(len(shuffled_idx)*0.8)]
test_idx = shuffled_idx[int(len(shuffled_idx)*0.8):]
test_normal_prep_eegs, test_normal_imgs, test_normal_files = \
    train_prep_eegs[test_idx], train_imgs[test_idx], train_files[test_idx]
train_prep_eegs, train_imgs, train_files = \
    train_prep_eegs[train_idx], train_imgs[train_idx], train_files[train_idx]

# Train data loader
train_imgs = torch.FloatTensor(train_imgs)
train_loader = torch.utils.data.DataLoader(train_imgs, batch_size=batch_size, shuffle=True)

if Restore:
    # Load test EEG data
    with open('{}_seizure_eegs.pickle'.format(modality), 'rb') as handle:
        test_seizure_eegs = pickle.load(handle)
    with open('{}_seizure_files.pickle'.format(modality), 'rb') as handle:
        test_seizure_files = pickle.load(handle)
    test_seizure_prep_eegs, test_seizure_files = apply_sliding_window(test_seizure_files, test_seizure_eegs)
    test_seizure_imgs = np.array([(img - np.min(img)) / (np.max(img) - np.min(img))
                          for img in test_seizure_prep_eegs])[:, np.newaxis, :, :]  # batch x 1 x channels x time points
    with open('{}_pd_eegs.pickle'.format(modality), 'rb') as handle:
        test_pd_eegs = pickle.load(handle)
    with open('{}_pd_files.pickle'.format(modality), 'rb') as handle:
        test_pd_files = pickle.load(handle)
    test_pd_prep_eegs, test_pd_files = apply_sliding_window(test_pd_files, test_pd_eegs)
    test_pd_imgs = np.array([(img - np.min(img)) / (np.max(img) - np.min(img))
                          for img in test_pd_prep_eegs])[:, np.newaxis, :, :]  # batch x 1 x channels x time points
    with open('{}_rda_eegs.pickle'.format(modality), 'rb') as handle:
        test_rd_eegs = pickle.load(handle)
    with open('{}_rda_files.pickle'.format(modality), 'rb') as handle:
        test_rd_files = pickle.load(handle)
    test_rd_prep_eegs, test_rd_files = apply_sliding_window(test_rd_files, test_rd_eegs)
    test_rd_imgs = np.array([(img - np.min(img)) / (np.max(img) - np.min(img))
                          for img in test_rd_prep_eegs])[:, np.newaxis, :, :]  # batch x 1 x channels x time points
    print("Number of test normal, seizure, pd, rd signals:", len(test_normal_imgs), len(test_seizure_imgs),
          len(test_pd_imgs), len(test_rd_imgs))  # 24199 4836 17953 6667

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.econv1 = nn.Conv2d(1, h_layer_1, kernel_size=kernel_size, stride=stride)
        self.ebn1 = nn.BatchNorm2d(h_layer_1, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)
        self.econv2 = nn.Conv2d(h_layer_1, h_layer_2, kernel_size=kernel_size, stride=stride)
        self.ebn2 = nn.BatchNorm2d(h_layer_2, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)
        self.efc1  = nn.Linear(h_layer_2 * (n_channels-6) * (sub_window_size-6), h_layer_5)
        self.edrop1 = nn.Dropout(p=0.3, inplace = False)
        self.mu_z  = nn.Linear(h_layer_5, latent_dim)
        self.scale_z = nn.Linear(h_layer_5, latent_dim)
        #
        self.dfc1 = nn.Linear(latent_dim, h_layer_5)
        self.dfc2 = nn.Linear(h_layer_5, h_layer_2 * (n_channels-6) * (sub_window_size-6))
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
        eh5 = self.relu(self.edrop1(self.efc1(eh2.view(-1, h_layer_2 * (n_channels-6) * (sub_window_size-6)))))
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
        dh5 = self.relu(self.dbn3(self.dconv3(dh2.view(-1, h_layer_2, (n_channels-6), (sub_window_size-6)))))
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
    # resume from checkpoint
    if checkpoint_epoch > 0:
        vae.load_state_dict(torch.load(PATH_vae + "_ep_{}".format(checkpoint_epoch + 1), map_location=lambda storage, loc: storage))
    for i in np.arange(checkpoint_epoch, epoch_num):
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

    # Load if results are already saved
    if os.path.exists(results_save_dir + '/anom_scores_normal_l_{}_input_{}.npy'.format(latent_dim, img_size)) and \
        os.path.exists(results_save_dir + '/anom_scores_seizure_l_{}_input_{}.npy'.format(latent_dim, img_size)) and \
        os.path.exists(results_save_dir + '/anom_scores_pd_l_{}_input_{}.npy'.format(latent_dim, img_size)) and \
        os.path.exists(results_save_dir + '/anom_scores_rd_l_{}_input_{}.npy'.format(latent_dim, img_size)) and \
            os.path.exists(results_save_dir + '/latent_tsne_l_{}_input_{}.npy'.format(latent_dim, img_size)):
        anom_scores_normal = np.load(results_save_dir + '/anom_scores_normal_l_{}_input_{}.npy'.
                                     format(latent_dim, img_size))
        anom_scores_seizure = np.load(results_save_dir + '/anom_scores_seizure_l_{}_input_{}.npy'.
                                      format(latent_dim, img_size))
        anom_scores_pd = np.load(results_save_dir + '/anom_scores_pd_l_{}_input_{}.npy'.
                                      format(latent_dim, img_size))
        anom_scores_rd = np.load(results_save_dir + '/anom_scores_rd_l_{}_input_{}.npy'.
                                      format(latent_dim, img_size))
        latent_vars_embedded = np.load(results_save_dir + '/latent_tsne_l_{}_input_{}.npy'.
                                               format(latent_dim, img_size))
    else:
        latent_vars_normal = []  # batch x latent_dim
        latent_vars_seizure = []  # batch x latent_dim
        latent_vars_pd = []  # batch x latent_dim
        latent_vars_rd = []  # batch x latent_dim
        anom_scores_normal = []  # batch x n_channels x sub_window_size
        anom_scores_seizure = []  # batch x n_channels x sub_window_size
        anom_scores_pd = []  # batch x n_channels x sub_window_size
        anom_scores_rd = []  # batch x n_channels x sub_window_size

        for file_name, img in zip(test_normal_files, test_normal_imgs):
            img_variable = Variable(torch.FloatTensor(img))
            img_variable = img_variable.unsqueeze(0)
            img_variable = img_variable.to(device)
            test_imgs_z_mu, test_imgs_z_scale = vae.Encoder(img_variable)
            latent_vars_normal.append(test_imgs_z_mu.squeeze(0).squeeze(0).cpu().data.numpy())
            # Repeat and average reconstruction
            test_imgs_rec = []  # img_size vector
            for _ in range(5):
                test_imgs_z = vae.Reparam(test_imgs_z_mu, test_imgs_z_scale)
                test_imgs_rec.append(vae.Decoder(test_imgs_z).squeeze(0).squeeze(0).cpu().data.numpy())
            # Use reconstruction error as anomaly score
            img_i = np.mean(test_imgs_rec, 0).reshape(n_channels, sub_window_size)
            img = img.reshape(n_channels, sub_window_size)
            anom_scores_normal.append(np.abs(img - img_i))

        for file_name, img in zip(test_seizure_files, test_seizure_imgs):
            img_variable = Variable(torch.FloatTensor(img))
            img_variable = img_variable.unsqueeze(0)
            img_variable = img_variable.to(device)
            test_imgs_z_mu, test_imgs_z_scale = vae.Encoder(img_variable)
            latent_vars_seizure.append(test_imgs_z_mu.squeeze(0).squeeze(0).cpu().data.numpy())
            # Repeat and average reconstruction
            test_imgs_rec = []  # img_size vector
            for _ in range(5):
                test_imgs_z = vae.Reparam(test_imgs_z_mu, test_imgs_z_scale)
                test_imgs_rec.append(vae.Decoder(test_imgs_z).squeeze(0).squeeze(0).cpu().data.numpy())
            # Use reconstruction error as anomaly score
            img_i = np.mean(test_imgs_rec, 0).reshape(n_channels, sub_window_size)
            img = img.reshape(n_channels, sub_window_size)
            anom_scores_seizure.append(np.abs(img - img_i))

        for file_name, img in zip(test_pd_files, test_pd_imgs):
            img_variable = Variable(torch.FloatTensor(img))
            img_variable = img_variable.unsqueeze(0)
            img_variable = img_variable.to(device)
            test_imgs_z_mu, test_imgs_z_scale = vae.Encoder(img_variable)
            latent_vars_pd.append(test_imgs_z_mu.squeeze(0).squeeze(0).cpu().data.numpy())
            # Repeat and average reconstruction
            test_imgs_rec = []  # img_size vector
            for _ in range(5):
                test_imgs_z = vae.Reparam(test_imgs_z_mu, test_imgs_z_scale)
                test_imgs_rec.append(vae.Decoder(test_imgs_z).squeeze(0).squeeze(0).cpu().data.numpy())
            # Use reconstruction error as anomaly score
            img_i = np.mean(test_imgs_rec, 0).reshape(n_channels, sub_window_size)
            img = img.reshape(n_channels, sub_window_size)
            anom_scores_pd.append(np.abs(img - img_i))

        for file_name, img in zip(test_rd_files, test_rd_imgs):
            img_variable = Variable(torch.FloatTensor(img))
            img_variable = img_variable.unsqueeze(0)
            img_variable = img_variable.to(device)
            test_imgs_z_mu, test_imgs_z_scale = vae.Encoder(img_variable)
            latent_vars_rd.append(test_imgs_z_mu.squeeze(0).squeeze(0).cpu().data.numpy())
            # Repeat and average reconstruction
            test_imgs_rec = []  # img_size vector
            for _ in range(5):
                test_imgs_z = vae.Reparam(test_imgs_z_mu, test_imgs_z_scale)
                test_imgs_rec.append(vae.Decoder(test_imgs_z).squeeze(0).squeeze(0).cpu().data.numpy())
            # Use reconstruction error as anomaly score
            img_i = np.mean(test_imgs_rec, 0).reshape(n_channels, sub_window_size)
            img = img.reshape(n_channels, sub_window_size)
            anom_scores_rd.append(np.abs(img - img_i))

        np.save(results_save_dir + '/anom_scores_normal_l_{}_input_{}.npy'.
                format(latent_dim, img_size), anom_scores_normal)
        np.save(results_save_dir + '/anom_scores_seizure_l_{}_input_{}.npy'.
                format(latent_dim, img_size), anom_scores_seizure)
        np.save(results_save_dir + '/anom_scores_pd_l_{}_input_{}.npy'.
                format(latent_dim, img_size), anom_scores_pd)
        np.save(results_save_dir + '/anom_scores_rd_l_{}_input_{}.npy'.
                format(latent_dim, img_size), anom_scores_rd)

        # Dimension reduction on latent space
        latent_vars = np.concatenate([latent_vars_normal, latent_vars_seizure], 0)
        print("Latent space matrix original shape:", latent_vars.shape)
        if latent_dim > 3:
            latent_vars_embedded_seizure = TSNE(n_components=3).fit_transform(latent_vars)
        else:
            latent_vars_embedded_seizure = np.copy(latent_vars)
        latent_vars = np.concatenate([latent_vars_normal, latent_vars_pd], 0)
        print("Latent space matrix original shape:", latent_vars.shape)
        if latent_dim > 3:
            latent_vars_embedded_pd = TSNE(n_components=3).fit_transform(latent_vars)
        else:
            latent_vars_embedded_pd = np.copy(latent_vars)
        latent_vars = np.concatenate([latent_vars_normal, latent_vars_rd], 0)
        print("Latent space matrix original shape:", latent_vars.shape)
        if latent_dim > 3:
            latent_vars_embedded_rd = TSNE(n_components=3).fit_transform(latent_vars)
        else:
            latent_vars_embedded_rd = np.copy(latent_vars)
        latent_vars_embedded = np.concatenate([latent_vars_embedded_seizure,
                            latent_vars_embedded_pd[len(latent_vars_normal):],
                            latent_vars_embedded_rd[len(latent_vars_normal):]], 0)
        np.save(results_save_dir + '/latent_tsne_l_{}_input_{}.npy'.format(latent_dim, img_size), latent_vars_embedded)
        print("Latent space matrix reduced shape:", latent_vars_embedded.shape)

    # Plot 3D latent space w.r.t. only categories
    plt.figure()
    _, ax = plt.subplots(1, 3)
    ax[0].scatter(latent_vars_embedded[:len(test_normal_files), 0], latent_vars_embedded[:len(test_normal_files), 1],
                     c="b", label="Normal", alpha=0.5)
    ax[0].scatter(latent_vars_embedded[len(test_normal_files):len(test_normal_files)+len(test_seizure_files), 0],
                 latent_vars_embedded[len(test_normal_files):len(test_normal_files)+len(test_seizure_files):, 1],
                 c="r", label="Seizure", alpha=0.5)
    ax[0].set(xlabel="x vs. y")
    ax[1].scatter(latent_vars_embedded[:len(test_normal_files), 0], latent_vars_embedded[:len(test_normal_files), 2],
                     c="b", label="Normal", alpha=0.5)
    ax[1].scatter(latent_vars_embedded[len(test_normal_files):len(test_normal_files) + len(test_seizure_files), 0],
                     latent_vars_embedded[len(test_normal_files):len(test_normal_files) + len(test_seizure_files):, 2],
                     c="r", label="Seizure", alpha=0.5)
    ax[1].set(xlabel="x vs. z")
    ax[2].scatter(latent_vars_embedded[:len(test_normal_files), 1], latent_vars_embedded[:len(test_normal_files), 2],
                     c="b", label="Normal", alpha=0.5)
    ax[2].scatter(latent_vars_embedded[len(test_normal_files):len(test_normal_files) + len(test_seizure_files), 1],
                     latent_vars_embedded[len(test_normal_files):len(test_normal_files) + len(test_seizure_files):, 2],
                     c="r", label="Seizure", alpha=0.5)
    ax[2].set(xlabel="y vs. z")
    plt.title("Normal vs. Seizure")
    plt.savefig(results_save_dir + '/latent_seizure_3D_l_{}_input_{}.pdf'.format(latent_dim, img_size), bbox_inches='tight')
    plt.close()

    plt.figure()
    _, ax = plt.subplots(1, 3)
    ax[0].scatter(latent_vars_embedded[:len(test_normal_files), 0], latent_vars_embedded[:len(test_normal_files), 1],
                     c="b", label="Normal", alpha=0.5)
    ax[0].scatter(latent_vars_embedded[len(test_normal_files) + len(test_seizure_files):
                                          len(test_normal_files) + len(test_seizure_files) + len(test_pd_files), 0],
                     latent_vars_embedded[len(test_normal_files) + len(test_seizure_files):
                                          len(test_normal_files) + len(test_seizure_files) + len(test_pd_files), 1],
                     c="g", label="PD", alpha=0.5)
    ax[0].set(xlabel="x vs. y")
    ax[1].scatter(latent_vars_embedded[:len(test_normal_files), 0], latent_vars_embedded[:len(test_normal_files), 2],
                     c="b", label="Normal", alpha=0.5)
    ax[1].scatter(latent_vars_embedded[len(test_normal_files) + len(test_seizure_files):
                                          len(test_normal_files) + len(test_seizure_files) + len(test_pd_files), 0],
                     latent_vars_embedded[len(test_normal_files) + len(test_seizure_files):
                                          len(test_normal_files) + len(test_seizure_files) + len(test_pd_files), 2],
                     c="g", label="PD", alpha=0.5)
    ax[1].set(xlabel="x vs. z")
    ax[2].scatter(latent_vars_embedded[:len(test_normal_files), 1], latent_vars_embedded[:len(test_normal_files), 2],
                     c="b", label="Normal", alpha=0.5)
    ax[2].scatter(latent_vars_embedded[len(test_normal_files) + len(test_seizure_files):
                                          len(test_normal_files) + len(test_seizure_files) + len(test_pd_files), 1],
                     latent_vars_embedded[len(test_normal_files) + len(test_seizure_files):
                                          len(test_normal_files) + len(test_seizure_files) + len(test_pd_files), 2],
                     c="g", label="PD", alpha=0.5)
    ax[2].set(xlabel="y vs. z")
    plt.title("Normal vs. PD")
    plt.savefig(results_save_dir + '/latent_pd_3D_l_{}_input_{}.pdf'.format(latent_dim, img_size), bbox_inches='tight')
    plt.close()

    plt.figure()
    _, ax = plt.subplots(1, 3)
    ax[0].scatter(latent_vars_embedded[:len(test_normal_files), 0], latent_vars_embedded[:len(test_normal_files), 1],
                     c="b", label="Normal", alpha=0.5)
    ax[0].scatter(latent_vars_embedded[-len(test_rd_files):, 0], latent_vars_embedded[-len(test_rd_files):, 1],
                     c="k", label="RD", alpha=0.5)
    ax[0].set(xlabel="x vs. y")
    ax[1].scatter(latent_vars_embedded[:len(test_normal_files), 0], latent_vars_embedded[:len(test_normal_files), 2],
                     c="b", label="Normal", alpha=0.5)
    ax[1].scatter(latent_vars_embedded[-len(test_rd_files):, 0], latent_vars_embedded[-len(test_rd_files):, 2],
                     c="k", label="RD", alpha=0.5)
    ax[1].set(xlabel="x vs. z")
    ax[2].scatter(latent_vars_embedded[:len(test_normal_files), 1], latent_vars_embedded[:len(test_normal_files), 2],
                     c="b", label="Normal", alpha=0.5)
    ax[2].scatter(latent_vars_embedded[-len(test_rd_files):, 1], latent_vars_embedded[-len(test_rd_files):, 2],
                     c="k", label="RD", alpha=0.5)
    ax[2].set(xlabel="y vs. z")
    plt.title("Normal vs. RD")
    plt.savefig(results_save_dir + '/latent_rd_3D_l_{}_input_{}.pdf'.format(latent_dim, img_size), bbox_inches='tight')
    plt.close()

    # Median over time to combat noise artifacts, max over time since only one channel may have biomarker
    anom_scores = np.concatenate([anom_scores_normal, anom_scores_seizure,
                                  anom_scores_pd, anom_scores_rd], 0)  # batch x n_channels x sub_window_size
    anom_avg_scores = np.median(anom_scores, -1)
    detect_channels = np.argmax(anom_avg_scores, -1)
    anom_avg_scores = np.max(anom_avg_scores, -1)

    # Plot anomaly score histograms w.r.t. only categories
    plt.figure()
    _, ax = plt.subplots()
    plt.hist(anom_avg_scores[:len(anom_scores_normal)], 50, density=True, facecolor="b", label="Normal", alpha=0.5)
    plt.hist(anom_avg_scores[len(anom_scores_normal):len(anom_scores_normal) + len(anom_scores_seizure)],
             50, density=True, facecolor="r", label="Seizure", alpha=0.5)
    ax.legend()
    ax.set(xlabel='Biomarker Evidence Score [0,1]')
    plt.savefig(results_save_dir + '/anom_hist_l_{}_input_{}.pdf'.format(latent_dim, img_size), bbox_inches='tight')
    plt.close()

    plt.figure()
    _, ax = plt.subplots()
    plt.hist(anom_avg_scores[:len(anom_scores_normal)], 50, density=True, facecolor="b", label="Normal", alpha=0.5)
    plt.hist(anom_avg_scores[len(anom_scores_normal):len(anom_scores_normal)+len(anom_scores_seizure)],
             50, density=True, facecolor="r", label="Seizure", alpha=0.5)
    plt.hist(anom_avg_scores[len(anom_scores_normal) + len(anom_scores_seizure):
                             len(anom_scores_normal) + len(anom_scores_seizure) + len(anom_scores_pd)],
             50, density=True, facecolor="g", label="PD", alpha=0.5)
    plt.hist(anom_avg_scores[-len(anom_scores_rd):],
             50, density=True, facecolor="k", label="RD", alpha=0.5)
    ax.legend()
    ax.set(xlabel='Biomarker Evidence Score [0,1]')
    plt.savefig(results_save_dir + '/anom_hist_all_l_{}_input_{}.pdf'.format(latent_dim, img_size), bbox_inches='tight')
    plt.close()
    #print("P-value between normal and seizure:", ttest_ind(anom_avg_scores[:len(anom_scores_normal)],
    #                                    anom_avg_scores[len(anom_scores_normal):], equal_var=False))

    # Test on normal and seizure windows
    # Each time window over all channels has one label. Max over channels, median over window
    test_labels = np.array([0] * len(anom_scores_normal) + [1] * len(anom_scores_seizure))
    anom_avg_scores_normal_seizure = np.concatenate([anom_avg_scores[:len(test_normal_files)],
                        anom_avg_scores[len(test_normal_files):len(test_normal_files) + len(test_seizure_files)]], 0)
    # Choose classification threshold
    auc = roc_auc_score(test_labels, anom_avg_scores_normal_seizure)
    fpr, tpr, thresholds = roc_curve(test_labels, anom_avg_scores_normal_seizure)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    print('Normal vs. Seizure classification threshold=%f' % (thresholds[ix]))
    anom_avg_scores_thresholded = np.array(anom_avg_scores_normal_seizure > thresholds[ix])
    report_dict = classification_report(test_labels, anom_avg_scores_thresholded, output_dict=True)
    precision = (report_dict["macro avg"]["precision"])
    recall = (report_dict["macro avg"]["recall"])
    accuracy = (report_dict["accuracy"])
    print("Normal vs. Seizure precision, recall, accuracy, AUC", precision, recall, accuracy, auc)

    # Sample plots for tp, fp, fn, tn
    T = np.arange(1, sub_window_size + 1) / (fs / downsample_factor)
    true_positive_idx = [idx for idx in range(len(test_labels)) if
                         test_labels[idx] == 1 and anom_avg_scores_thresholded[idx] == 1]
    false_positive_idx = [idx for idx in range(len(test_labels)) if
                         test_labels[idx] == 0 and anom_avg_scores_thresholded[idx] == 1]
    true_negative_idx = [idx for idx in range(len(test_labels)) if
                         test_labels[idx] == 0 and anom_avg_scores_thresholded[idx] == 0]
    false_negative_idx = [idx for idx in range(len(test_labels)) if
                         test_labels[idx] == 1 and anom_avg_scores_thresholded[idx] == 0]
    all_test_prep_eegs = np.concatenate([test_normal_prep_eegs, test_seizure_prep_eegs], 0)
    all_test_files = np.concatenate([test_normal_files, test_seizure_files], 0)

    plt.figure()
    _, axs = plt.subplots(2, 2)
    vis_idx = 0
    plotted_patient_names = []
    for window_idx in true_positive_idx:
        # Plot original unnormalized signal
        img = all_test_prep_eegs[window_idx]
        patient_name = all_test_files[window_idx].split("(")[1].split(")")[0]
        ch = detect_channels[window_idx]
        if patient_name not in plotted_patient_names or len(plotted_patient_names) % 2 == 1:  # plot at least 2 patients
            plotted_patient_names.append(patient_name)
            axs[int(vis_idx / 2), int(vis_idx % 2)].plot(T, img[ch], c="b", linewidth=0.5)
            axs[int(vis_idx / 2), int(vis_idx % 2)].set(title="Patient {} from site {}".format(
                patient_name.split("_")[2], patient_name.split("_")[1]))
            vis_idx += 1
            if vis_idx == 4:
                break
    axs[0, 0].set(ylabel=r'$\mu V$')
    axs[0, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # hide x ticks for top 2
    axs[0, 1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # hide x ticks for top 2
    axs[1, 0].set(ylabel=r'$\mu V$')
    axs[1, 0].set(xlabel=r'Time (s)')
    axs[1, 1].set(xlabel=r'Time (s)')
    plt.savefig(results_save_dir + '/true_positive_l_{}_input_{}.pdf'.format(latent_dim, img_size),
                bbox_inches='tight')
    plt.close()

    plt.figure()
    _, axs = plt.subplots(2, 2)
    vis_idx = 0
    plotted_patient_names = []
    for window_idx in false_positive_idx:
        # Plot original unnormalized signal
        img = all_test_prep_eegs[window_idx]
        patient_name = all_test_files[window_idx].split("(")[1].split(")")[0]
        ch = detect_channels[window_idx]
        if patient_name not in plotted_patient_names or len(plotted_patient_names) % 2 == 1:  # plot at least 2 patients
            plotted_patient_names.append(patient_name)
            axs[int(vis_idx / 2), int(vis_idx % 2)].plot(T, img[ch], c="b", linewidth=0.5)
            axs[int(vis_idx / 2), int(vis_idx % 2)].set(title="Patient {} from site {}".format(
                patient_name.split("_")[2], patient_name.split("_")[1]))
            vis_idx += 1
            if vis_idx == 4:
                break
    axs[0, 0].set(ylabel=r'$\mu V$')
    axs[0, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # hide x ticks for top 2
    axs[0, 1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # hide x ticks for top 2
    axs[1, 0].set(ylabel=r'$\mu V$')
    axs[1, 0].set(xlabel=r'Time (s)')
    axs[1, 1].set(xlabel=r'Time (s)')
    plt.savefig(results_save_dir + '/false_positive_l_{}_input_{}.pdf'.format(latent_dim, img_size),
                bbox_inches='tight')
    plt.close()

    plt.figure()
    _, axs = plt.subplots(2, 2)
    vis_idx = 0
    plotted_patient_names = []
    for window_idx in true_negative_idx:
        # Plot original unnormalized signal
        img = all_test_prep_eegs[window_idx]
        patient_name = all_test_files[window_idx].split("(")[1].split(")")[0]
        ch = detect_channels[window_idx]
        if patient_name not in plotted_patient_names or len(plotted_patient_names) % 2 == 1:  # plot at least 2 patients
            plotted_patient_names.append(patient_name)
            axs[int(vis_idx / 2), int(vis_idx % 2)].plot(T, img[ch], c="b", linewidth=0.5)
            axs[int(vis_idx / 2), int(vis_idx % 2)].set(title="Patient {} from site {}".format(
                patient_name.split("_")[2], patient_name.split("_")[1]))
            vis_idx += 1
            if vis_idx == 4:
                break
    axs[0, 0].set(ylabel=r'$\mu V$')
    axs[0, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # hide x ticks for top 2
    axs[0, 1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # hide x ticks for top 2
    axs[1, 0].set(ylabel=r'$\mu V$')
    axs[1, 0].set(xlabel=r'Time (s)')
    axs[1, 1].set(xlabel=r'Time (s)')
    plt.savefig(results_save_dir + '/true_negative_l_{}_input_{}.pdf'.format(latent_dim, img_size),
                bbox_inches='tight')
    plt.close()

    plt.figure()
    _, axs = plt.subplots(2, 2)
    vis_idx = 0
    plotted_patient_names = []
    for window_idx in false_negative_idx:
        # Plot original unnormalized signal
        img = all_test_prep_eegs[window_idx]
        patient_name = all_test_files[window_idx].split("(")[1].split(")")[0]
        ch = detect_channels[window_idx]
        if patient_name not in plotted_patient_names or len(plotted_patient_names) % 2 == 1:  # plot at least 2 patients
            plotted_patient_names.append(patient_name)
            axs[int(vis_idx / 2), int(vis_idx % 2)].plot(T, img[ch], c="b", linewidth=0.5)
            axs[int(vis_idx / 2), int(vis_idx % 2)].set(title="Patient {} from site {}".format(
                patient_name.split("_")[2], patient_name.split("_")[1]))
            vis_idx += 1
            if vis_idx == 4:
                break
    axs[0, 0].set(ylabel=r'$\mu V$')
    axs[0, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # hide x ticks for top 2
    axs[0, 1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # hide x ticks for top 2
    axs[1, 0].set(ylabel=r'$\mu V$')
    axs[1, 0].set(xlabel=r'Time (s)')
    axs[1, 1].set(xlabel=r'Time (s)')
    plt.savefig(results_save_dir + '/false_negative_l_{}_input_{}.pdf'.format(latent_dim, img_size),
                bbox_inches='tight')
    plt.close()

    # Test on normal and PD windows
    # Each time window over all channels has one label. Max over channels, median over window
    test_labels = np.array([0] * len(anom_scores_normal) + [1] * len(anom_scores_pd))
    anom_avg_scores_normal_pd = np.concatenate([anom_avg_scores[:len(test_normal_files)],
                                anom_avg_scores[len(test_normal_files) + len(test_seizure_files):
                                len(test_normal_files) + len(test_seizure_files) + len(test_pd_files)]], 0)
    # Choose classification threshold
    auc = roc_auc_score(test_labels, anom_avg_scores_normal_pd)
    fpr, tpr, thresholds = roc_curve(test_labels, anom_avg_scores_normal_pd)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    print('Normal vs. PD classification threshold=%f' % (thresholds[ix]))
    anom_avg_scores_thresholded = np.array(anom_avg_scores_normal_pd > thresholds[ix])
    report_dict = classification_report(test_labels, anom_avg_scores_thresholded, output_dict=True)
    precision = (report_dict["macro avg"]["precision"])
    recall = (report_dict["macro avg"]["recall"])
    accuracy = (report_dict["accuracy"])
    print("Normal vs. PD precision, recall, accuracy, AUC", precision, recall, accuracy, auc)

    # Test on normal and RD windows
    # Each time window over all channels has one label. Max over channels, median over window
    test_labels = np.array([0] * len(anom_scores_normal) + [1] * len(anom_scores_rd))
    anom_avg_scores_normal_rd = np.concatenate([anom_avg_scores[:len(test_normal_files)],
                                                anom_avg_scores[-len(test_rd_files):]], 0)
    # Choose classification threshold
    auc = roc_auc_score(test_labels, anom_avg_scores_normal_rd)
    fpr, tpr, thresholds = roc_curve(test_labels, anom_avg_scores_normal_rd)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    print('Normal vs. RD classification threshold=%f' % (thresholds[ix]))
    anom_avg_scores_thresholded = np.array(anom_avg_scores_normal_rd > thresholds[ix])
    report_dict = classification_report(test_labels, anom_avg_scores_thresholded, output_dict=True)
    precision = (report_dict["macro avg"]["precision"])
    recall = (report_dict["macro avg"]["recall"])
    accuracy = (report_dict["accuracy"])
    print("Normal vs. RD precision, recall, accuracy, AUC", precision, recall, accuracy, auc)

if Restore:
    plot_reconstruction()