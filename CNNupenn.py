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
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
from scipy.signal import spectrogram
from scipy.stats import ttest_ind
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, classification_report
import matplotlib

Restore = False
modality = "upenn_extended"
fs = 500.0
img_size = 36000  # to cover shortest activity of 1 s
n_channels = 72
sub_window_size = int(img_size / n_channels)  # sub_window_size / fs second window
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
            signal_per_channel_sliding = extract_windows_vectorized(signal[channel_index], sub_window_size,
                                                                    downsample_factor)
            current_signals.append(signal_per_channel_sliding)
        # replicate through channels if there are less channels than max n_channels
        current_signals = np.array(current_signals)
        print("Sliding signal shape channels x batch x time points:", current_signals.shape)
        if signal.shape[0] < n_channels:
            current_signals = np.tile(current_signals,
                                      [int(np.ceil(n_channels / signal.shape[0])), 1, 1])
            current_signals = current_signals[:n_channels]
        # batch x channels x time points
        current_signals = current_signals.transpose((1, 0, 2))
        print("Sliding output signal shape batch x channels x time points:", current_signals.shape)
        current_file_names = np.tile([file_name], (len(current_signals),))
        prep_eegs.extend(current_signals)
        prep_files.extend(current_file_names)
    prep_eegs = np.array(prep_eegs)
    print("Dataset shape:", prep_eegs.shape)
    prep_files = np.array(prep_files)
    return prep_eegs, prep_files

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(' Processor is %s' % (device))

# model parameters
h_layer_1 = 32
h_layer_2 = 16
h_layer_3 = 8
h_layer_4 = 1000
h_layer_5 = 500
h_layer_6 = 64
kernel_size = 4
stride = 1

# training parameters
batch_size = 32
epoch_num = 200
learning_rate = 1e-3

# Save folders for trained models and logs
model_save_dir = "models_{}_CNN".format(modality)
if not os.path.isdir(model_save_dir):
    os.mkdir(model_save_dir)
results_save_dir = "results_{}_CNN".format(modality)
if not os.path.isdir(results_save_dir):
    os.mkdir(results_save_dir)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.econv1 = nn.Conv1d(n_channels, h_layer_1, kernel_size=kernel_size, stride=stride)
        self.ebn1 = nn.BatchNorm1d(h_layer_1, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)
        self.econv2 = nn.Conv1d(h_layer_1, h_layer_2, kernel_size=kernel_size, stride=stride)
        self.ebn2 = nn.BatchNorm1d(h_layer_2, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)
        self.econv3 = nn.Conv1d(h_layer_2, h_layer_3, kernel_size=kernel_size, stride=stride)
        self.ebn3 = nn.BatchNorm1d(h_layer_3, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)

        self.efc1 = nn.Linear(h_layer_3 * (sub_window_size-18), h_layer_4)
        self.efc2 = nn.Linear(h_layer_4, h_layer_5)
        self.efc3 = nn.Linear(h_layer_5, 2)
        self.drop = nn.Dropout(p=0.3, inplace=False)
        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, x):
        eh1 = self.pool(self.drop(self.relu(self.ebn1(self.econv1(x)))))
        eh2 = self.pool(self.drop(self.relu(self.ebn2(self.econv2(eh1)))))
        eh3 = self.pool(self.drop(self.relu(self.ebn3(self.econv3(eh2)))))
        #print(eh3.shape)
        eh4 = self.drop(self.relu(self.efc1(eh3.view(-1, h_layer_3 * (sub_window_size-18)))))
        eh5 = self.drop(self.relu(self.efc2(eh4)))
        pred = self.softmax(self.efc3(eh5))
        return pred

# Load normal EEG data
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
normal_prep_eegs, normal_files = apply_sliding_window(train_files_cleaned, train_eegs_cleaned)

# Load seizure EEG data
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
seizure_prep_eegs, seizure_files = apply_sliding_window(test_seizure_files_cleaned, test_seizure_eegs_cleaned)

# Entire dataset
all_prep_eegs = np.concatenate([normal_prep_eegs, seizure_prep_eegs])
all_files = np.concatenate([normal_files, seizure_files])
all_labels = np.array([0] * len(normal_prep_eegs) + [1] * len(seizure_prep_eegs))

# load dataset via min-max normalization & add image channel dimension
all_imgs = np.array([(img - np.min(img)) / (np.max(img) - np.min(img))
                       for img in all_prep_eegs])
                        #[:, np.newaxis, :, :]  # batch x 1 x channels x time points
print("Dataset shape", all_imgs.shape)

# initialize model
model = CNN()
model.to(device)
print(model)
model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Class weights
weights = torch.Tensor([len(seizure_files), len(normal_files)]).cuda()
loss_function = nn.CrossEntropyLoss(weight=weights)

# separate normal signals into train and test portions
kf = KFold(n_splits=5, shuffle=True, random_state=999)
fold_idx = 0
for train_idx, test_idx in kf.split(range(len(all_imgs))):
    fold_idx += 1
    PATH_model = model_save_dir + '/model_fold_{}'.format(fold_idx)
    checkpoint_epoch = 0

    test_prep_eegs, test_imgs, test_files, test_labels = \
        all_prep_eegs[test_idx], all_imgs[test_idx], all_files[test_idx], all_labels[test_idx]
    train_prep_eegs, train_imgs, train_files, train_labels = \
        all_prep_eegs[train_idx], all_imgs[train_idx], all_files[train_idx], all_labels[train_idx]

    train_x = torch.from_numpy(train_imgs).float()
    train_y = torch.from_numpy(train_labels).long()
    test_x = torch.from_numpy(test_imgs).float()
    test_y = torch.from_numpy(test_labels).long()

    # Pytorch train and test set data loaders
    train = torch.utils.data.TensorDataset(train_x, train_y)
    test = torch.utils.data.TensorDataset(test_x, test_y)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    # training
    if not Restore:
        print("Training...")
        best_loss = 1e10
        # resume from checkpoint
        if checkpoint_epoch > 0:
            model.load_state_dict(torch.load(PATH_model + "_ep_{}".format(checkpoint_epoch + 1), map_location=lambda storage, loc: storage))
        for i in np.arange(checkpoint_epoch, epoch_num):
            time_start = time.time()
            loss_model_value = 0.0
            for images, labels in train_loader:
                image_model = Variable(images).to(device)
                y_true = Variable(labels).to(device)

                model_optimizer.zero_grad()
                pred = model.forward(image_model)
                loss_model = loss_function(pred, y_true)

                loss_model.backward()
                loss_model_value += loss_model.data
                model_optimizer.step()

            time_end = time.time()
            print('elapsed time (min) : %0.1f' % ((time_end-time_start)/60))
            current_loss = loss_model_value / len(train_loader.dataset)
            print('====> Epoch: %d Loss : %0.8f' % ((i + 1), current_loss))
            if current_loss < best_loss:
                best_loss = current_loss
                best_epoch = i + 1
                torch.save(model.state_dict(), PATH_model + "_best")
            if i % 10 == 9:
                torch.save(model.state_dict(), PATH_model + "_ep_{}".format(i + 1))
        print("Best loss and epoch:", best_loss, best_epoch)

    if Restore:
        model.load_state_dict(torch.load(PATH_model + "_best", map_location=lambda storage, loc: storage))
        model.eval()

        # Iterate through test dataset
        predicted = []
        all_labels = []
        predicted_probs = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = Variable(images).to(device)
                labels = labels.numpy()

                # Forward propagation
                outputs = model.forward(images).data.cpu().numpy()

                # Get predictions by thresholding
                predicted.extend(np.argmax(outputs, 1).tolist())
                predicted_probs.extend(outputs[:, 1].tolist())
                all_labels.extend(labels.tolist())

        report_dict = classification_report(all_labels, predicted, output_dict=True)
        precision = (report_dict["macro_avg"]["precision"])
        recall = (report_dict["macro_avg"]["recall"])
        accuracy = (report_dict["accuracy"])
        auc = roc_auc_score(all_labels, predicted_probs)

        # Print Epoch Loss, Test Accuracy and AUC
        print('Test Precision: {} Recall: {} Accuracy: {} '
              'AUC: {} \n'.format(precision, recall, accuracy, auc))
        torch.cuda.empty_cache()