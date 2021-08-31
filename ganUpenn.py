from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch import nn, optim
import os
import time
import pickle
from scipy.signal import spectrogram
from scipy.stats import ttest_ind
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, classification_report
import matplotlib

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
Restore = False
modality = "upenn_extended"
fs = 500.0
sub_window_size = 7500
downsample_factor = 1
print("window size {}".format(sub_window_size))

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
        # spectrogram for each channel
        print("Input signal shape channels x time points:", signal.shape)
        current_signals = []
        for channel_index in range(signal.shape[0]):
            signal_per_channel_sliding = extract_windows_vectorized(signal[channel_index], sub_window_size,
                                                                    downsample_factor)
            print("Sliding signal shape per channel batch x time points:", signal_per_channel_sliding.shape)
            for batch_index in range(signal_per_channel_sliding.shape[0]):
                _, _, spec_per_channel_sliding = spectrogram(signal_per_channel_sliding[batch_index], fs=fs, return_onesided=False)
                print("Sliding spec shape per channel (raw):", spec_per_channel_sliding.shape)
                # Take two sides of spectogram
                spec_per_channel_sliding_side_1 = spec_per_channel_sliding[:int(len(spec_per_channel_sliding)/2)]
                spec_per_channel_sliding_side_2 = spec_per_channel_sliding[int(len(spec_per_channel_sliding)/2):]
                spec_per_channel_sliding = np.stack([spec_per_channel_sliding_side_1, spec_per_channel_sliding_side_2], 0)
                print("Sliding spec shape per channel (two sided):", spec_per_channel_sliding.shape)
                # Add mean channel: 3 x f x t
                spec_per_channel_sliding = np.concatenate([spec_per_channel_sliding,
                                                    np.mean(spec_per_channel_sliding, 0)[np.newaxis, :, :]], 0)
                print("Sliding spec shape per channel (with mean):", spec_per_channel_sliding.shape)
                current_signals.append(spec_per_channel_sliding)
        current_file_names = np.tile([file_name], (len(current_signals),))
        prep_eegs.extend(current_signals)
        prep_files.extend(current_file_names)
    prep_eegs = np.array(prep_eegs)
    print("Dataset shape:", prep_eegs.shape)
    prep_files = np.array(prep_files)
    return prep_eegs, prep_files

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[index]
        if self.transform:
            x = self.transform(x)
        return x

    def __len__(self):
        return self.tensors.size(0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, ngpu=1):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu=1):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(' Processor is %s' % (device))

# Batch size during training
batch_size = 128
# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 50
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 100
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Save folders for trained models and logs
model_save_dir = "models_{}_GAN".format(modality)
if not os.path.isdir(model_save_dir):
    os.mkdir(model_save_dir)
PATH_GAN = model_save_dir + '/GAN'
results_save_dir = "results_{}_GAN".format(modality)
if not os.path.isdir(results_save_dir):
    os.mkdir(results_save_dir)

# Load train EEG data with overlapping sliding windows
with open('{}_train_eegs.pickle'.format(modality), 'rb') as handle:
    train_eegs = pickle.load(handle)
with open('{}_train_files.pickle'.format(modality), 'rb') as handle:
    train_files = pickle.load(handle)
# filter nans
train_eegs_cleaned = []
train_files_cleaned = []
count = 0
for idx in range(len(train_eegs)):
    if np.any(np.isnan(train_eegs[idx])):
        count += 1
    else:
        train_eegs_cleaned.append(train_eegs[idx])
        train_files_cleaned.append(train_files[idx])
print("{} train eegs are cleaned".format(count))
train_prep_eegs, train_files = apply_sliding_window(train_files_cleaned, train_eegs_cleaned)
# load dataset via min-max normalization & add image channel dimension
train_imgs = np.array([(img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5)
                       for img in train_prep_eegs])  # batch x channels x time points
print("Range of normalized images:", np.min(train_imgs), np.max(train_imgs))

# separate normal signals into train and test portions
shuffled_idx = range(len(train_imgs))
train_idx = shuffled_idx[:int(len(shuffled_idx)*0.8)]
test_idx = shuffled_idx[int(len(shuffled_idx)*0.8):]
test_normal_prep_eegs, test_normal_imgs, test_normal_files = \
    train_prep_eegs[test_idx], train_imgs[test_idx], train_files[test_idx]
train_prep_eegs, train_imgs, train_files = \
    train_prep_eegs[train_idx], train_imgs[train_idx], train_files[train_idx]

# resize transforms
transform_list = transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Train data loader
train_imgs = torch.FloatTensor(train_imgs)
train_ds = CustomTensorDataset(train_imgs, transform_list)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

if Restore:
    # Load test EEG data
    with open('{}_seizure_eegs.pickle'.format(modality), 'rb') as handle:
        test_seizure_eegs = pickle.load(handle)
    with open('{}_seizure_files.pickle'.format(modality), 'rb') as handle:
        test_seizure_files = pickle.load(handle)
    # filter nans
    test_seizure_eegs_cleaned = []
    test_seizure_files_cleaned = []
    count = 0
    for idx in range(len(test_seizure_eegs)):
        if np.any(np.isnan(test_seizure_eegs[idx])):
            count += 1
        else:
            test_seizure_eegs_cleaned.append(test_seizure_eegs[idx])
            test_seizure_files_cleaned.append(test_seizure_files[idx])
    print("{} seizure eegs are cleaned".format(count))
    test_seizure_prep_eegs, test_seizure_files = \
        apply_sliding_window(test_seizure_files_cleaned, test_seizure_eegs_cleaned)
    test_seizure_imgs = np.array([(img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5)
                          for img in test_seizure_prep_eegs])  # batch x channels x time points
    print("Number of test normal, seizure signals:", len(test_normal_imgs), len(test_seizure_imgs))  # 93301 5511

    # Test loaders
    test_normal_imgs = torch.FloatTensor(test_normal_imgs)
    test_normal_ds = CustomTensorDataset(test_normal_imgs, transform_list)
    test_normal_loader = torch.utils.data.DataLoader(test_normal_ds, batch_size=1, shuffle=False)

    test_seizure_imgs = torch.FloatTensor(test_seizure_imgs)
    test_seizure_ds = CustomTensorDataset(test_seizure_imgs, transform_list)
    test_seizure_loader = torch.utils.data.DataLoader(test_seizure_ds, batch_size=1, shuffle=False)

# Architecture
# Create the generator
netG = Generator().to(device)
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)
# Print the model
print(netG)
# Create the Discriminator
netD = Discriminator().to(device)
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)
# Print the model
print(netD)

# Initialize BCELoss function
criterion = nn.BCELoss()
# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(ngf, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.
# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

if not Restore:
    # Training Loop
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(train_loader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data.to(device)
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(train_loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                torch.save(netD.state_dict(), PATH_GAN + "_D")
                torch.save(netG.state_dict(), PATH_GAN + "_G")

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(train_loader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(results_save_dir + "/loss.png")

if Restore:
    netD.load_state_dict(torch.load(PATH_GAN + "_D", map_location=lambda storage, loc: storage))
    netG.load_state_dict(torch.load(PATH_GAN + "_G", map_location=lambda storage, loc: storage))

def plot_reconstruction():
    netD.eval()
    netG.eval()

    # Load if results are already saved
    if os.path.exists(results_save_dir + '/anom_scores_normal.npy') and \
        os.path.exists(results_save_dir + '/anom_scores_seizure.npy'):

        anom_scores_normal = np.load(results_save_dir + '/anom_scores_normal.npy')
        anom_scores_seizure = np.load(results_save_dir + '/anom_scores_seizure.npy')
    else:
        anom_scores_normal = []
        anom_scores_seizure = []

        for img_variable in test_normal_loader:
            # Format batch
            real_cpu = img_variable.to(device)
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            D_x = output.mean().item()

            # Generate batch of latent vectors
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            D_G_z = output.mean().item()

            ###########################Anomaly loss
            x_G_z = np.mean(np.abs(img_variable.squeeze(0).data.numpy() - fake.squeeze(0).data.cpu().numpy()))
            D_x_D_G_z = np.abs(D_x - D_G_z)
            anom_scores_normal.append(x_G_z + D_x_D_G_z)

        for img_variable in test_seizure_loader:
            # Format batch
            real_cpu = img_variable.to(device)
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            D_x = output.mean().item()

            # Generate batch of latent vectors
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            D_G_z = output.mean().item()

            ###########################Anomaly loss
            x_G_z = np.mean(np.abs(img_variable.squeeze(0).data.numpy() - fake.squeeze(0).data.cpu().numpy()))
            D_x_D_G_z = np.abs(D_x - D_G_z)
            anom_scores_seizure.append(x_G_z + D_x_D_G_z)

        np.save(results_save_dir + '/anom_scores_normal.npy', anom_scores_normal)
        np.save(results_save_dir + '/anom_scores_seizure.npy', anom_scores_seizure)

    # Median over time to combat noise artifacts, max over time since only one channel may have activity
    anom_avg_scores = np.concatenate([anom_scores_normal, anom_scores_seizure], 0)

    print("P-value between normal and seizure:", ttest_ind(anom_avg_scores[:len(anom_scores_normal)],
                                        anom_avg_scores[len(anom_scores_normal):], equal_var=False))

    # Test on normal and seizure windows
    # Each time window over all channels has one label. Max over channels, median over window
    test_labels = np.array([0] * len(anom_scores_normal) + [1] * len(anom_scores_seizure))
    # Choose classification threshold
    auc = roc_auc_score(test_labels, anom_avg_scores)
    fpr, tpr, thresholds = roc_curve(test_labels, anom_avg_scores)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    print('Normal vs. Seizure classification threshold=%f' % (thresholds[ix]))
    anom_avg_scores_thresholded = np.array(anom_avg_scores > thresholds[ix])
    report_dict = classification_report(test_labels, anom_avg_scores_thresholded, output_dict=True)
    precision = (report_dict["macro avg"]["precision"])
    recall = (report_dict["macro avg"]["recall"])
    accuracy = (report_dict["accuracy"])
    print("Normal vs. Seizure precision, recall, accuracy, AUC", precision, recall, accuracy, auc)

if Restore:
    plot_reconstruction()