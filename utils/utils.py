# -*- coding: utf-8 -*-
from numpy.random import seed
seed(1017)
from tensorflow import set_random_seed
set_random_seed(1017)


from glob import glob
import os
from pathlib import Path
from collections import OrderedDict

import mne
from mne.io import RawArray
from mne import read_evokeds, read_source_spaces, compute_covariance
from mne import channels, find_events, concatenate_raws
from mne import pick_types, viz, io, Epochs, create_info
from mne import pick_channels, concatenate_epochs
from mne.datasets import sample
from mne.simulation import simulate_sparse_stc, simulate_raw
from mne.channels import read_montage, make_standard_montage, read_custom_montage
from mne.time_frequency import tfr_morlet

import pandas as pd
pd.options.display.precision = 4
pd.options.display.max_columns = None

from numpy import genfromtxt
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (12,12)

import keras
from keras import regularizers
from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.layers import Flatten, Conv2D, MaxPooling2D, LSTM
from keras.layers import BatchNormalization, Conv3D, MaxPooling3D

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

sns.set_context('talk')
sns.set_style('white')


def load_openBCI_csv_as_raw(filename, sfreq=256., ch_ind=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],                         stim_ind=16, replace_ch_names=None, verbose=1):
    """Load CSV files into a Raw object.

    Args:
        filename (list): paths to CSV files to load

    Keyword Args:
        subject_nb (int or str): subject number. If 'all', load all
            subjects.
        session_nb (int or str): session number. If 'all', load all
            sessions.
        sfreq (float): EEG sampling frequency
        ch_ind (list): indices of the EEG channels to keep
        stim_ind (int): index of the stim channel
        replace_ch_names (dict or None): dictionary containing a mapping to
            rename channels. Useful when an external electrode was used.

    Returns:
        (mne.io.array.array.RawArray): loaded EEG
        """
    n_channel = len(ch_ind)
    raw = []
    
    for fname in filename:
        # read the file
        data = pd.read_csv(fname, index_col=0)
        # name of each channels
        ch_names = list(data.columns)[0:n_channel] + ['Stim']
        if replace_ch_names is not None:
            ch_names = [c if c not in replace_ch_names.keys()
                        else replace_ch_names[c] for c in ch_names]
        # type of each channels
        ch_types = ['eeg'] * n_channel + ['stim']
        # get data and exclude Aux channel
        data = data.values[:, ch_ind + [stim_ind]].T
        # convert in Volts (from uVolts)
        data[:-1] *= 1e-6
        montage = make_standard_montage('standard_1005')
        info = create_info(ch_names=ch_names, ch_types=ch_types,
                            sfreq=sfreq, verbose=verbose)
        rawi = RawArray(data=data, info=info, verbose=verbose)
        rawi.set_montage(montage, raise_if_subset=False, match_case=False)
        raw.append(rawi)
        # concatenate all raw objects
        raws = concatenate_raws(raw, verbose=verbose)
    return raws

def load_data(sfreq=256.,ch_ind=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 
              stim_ind=16, replace_ch_names=None, verbose=1):
    """Load CSV files from the /data directory into a Raw object.

    Args:
        data_dir (str): directory inside /data that contains the
            CSV files to load, e.g., 'auditory/P300'

    Keyword Args:
        subject_nb (int or str): subject number. If 'all', load all
            subjects.
        session_nb (int or str): session number. If 'all', load all
            sessions.
        sfreq (float): EEG sampling frequency
        ch_ind (list): indices of the EEG channels to keep
        stim_ind (int): index of the stim channel
        replace_ch_names (dict or None): dictionary containing a mapping to
            rename channels. Useful when an external electrode was used.

    Returns:
        (mne.io.array.array.RawArray): loaded EEG
    """
    '''if subject_nb == 'all':
        subject_nb = '*'
    if session_nb == 'all':
        session_nb = '*'
    '''
    recording_path = os.path.join(Path().absolute(), "data")
    filename = '*.txt'
    data_path = os.path.join(recording_path, filename)
    fnames = glob(data_path)
    print(len(fnames))
    return load_openBCI_csv_as_raw(fnames, sfreq=sfreq, ch_ind=ch_ind,
                                stim_ind=stim_ind,
                                replace_ch_names=replace_ch_names, verbose=verbose)

def plot_conditions(epochs, conditions=OrderedDict(), ci=97.5, n_boot=1000,
                    title='', palette=None, ylim=(-6, 6),
                    diff_waveform=(1, 2)):
    """Plot ERP conditions.

    Args:
        epochs (mne.epochs): EEG epochs

    Keyword Args:
        conditions (OrderedDict): dictionary that contains the names of the
            conditions to plot as keys, and the list of corresponding marker
            numbers as value. E.g.,

                conditions = {'Non-target': [0, 1],
                               'Target': [2, 3, 4]}

        ci (float): confidence interval in range [0, 100]
        n_boot (int): number of bootstrap samples
        title (str): title of the figure
        palette (list): color palette to use for conditions
        ylim (tuple): (ymin, ymax)
        diff_waveform (tuple or None): tuple of ints indicating which
            conditions to subtract for producing the difference waveform.
            If None, do not plot a difference waveform

    Returns:
        (matplotlib.figure.Figure): figure object
        (list of matplotlib.axes._subplots.AxesSubplot): list of axes
    """
    if isinstance(conditions, dict):
        conditions = OrderedDict(conditions)

    if palette is None:
        palette = sns.color_palette("hls", len(conditions) + 1)

    X = epochs.get_data() * 1e6
    times = epochs.times
    y = pd.Series(epochs.events[:, -1])

    fig, axes = plt.subplots(2, 2, figsize=[12, 6],
                             sharex=True, sharey=True)
    axes = [axes[1, 0], axes[0, 0], axes[0, 1], axes[1, 1]]

    for ch in range(4):
        for cond, color in zip(conditions.values(), palette):
            sns.tsplot(X[y.isin(cond), ch], time=times, color=color,
                       n_boot=n_boot, ci=ci, ax=axes[ch])

        if diff_waveform:
            diff = (np.nanmean(X[y == diff_waveform[1], ch], axis=0) -
                    np.nanmean(X[y == diff_waveform[0], ch], axis=0))
            axes[ch].plot(times, diff, color='k', lw=1)

        axes[ch].set_title(epochs.ch_names[ch])
        axes[ch].set_ylim(ylim)
        axes[ch].axvline(x=0, ymin=ylim[0], ymax=ylim[1], color='k',
                         lw=1, label='_nolegend_')

    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (uV)')
    axes[-1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude (uV)')

    if diff_waveform:
        legend = (['{} - {}'.format(diff_waveform[1], diff_waveform[0])] +
                  list(conditions.keys()))
    else:
        legend = conditions.keys()
    axes[-1].legend(legend)
    sns.despine()
    plt.tight_layout()

    if title:
        fig.suptitle(title, fontsize=20)

    return fig, axes


def plot_highlight_regions(x, y, hue, hue_thresh=0, xlabel='', ylabel='',
                           legend_str=()):
    """Plot a line with highlighted regions based on additional value.

    Plot a line and highlight ranges of x for which an additional value
    is lower than a threshold. For example, the additional value might be
    pvalues, and the threshold might be 0.05.

    Args:
        x (array_like): x coordinates
        y (array_like): y values of same shape as `x`

    Keyword Args:
        hue (array_like): values to be plotted as hue based on `hue_thresh`.
            Must be of the same shape as `x` and `y`.
        hue_thresh (float): threshold to be applied to `hue`. Regions for which
            `hue` is lower than `hue_thresh` will be highlighted.
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        legend_str (tuple): legend for the line and the highlighted regions

    Returns:
        (matplotlib.figure.Figure): figure object
        (list of matplotlib.axes._subplots.AxesSubplot): list of axes
    """
    fig, axes = plt.subplots(1, 1, figsize=(10, 5), sharey=True)

    axes.plot(x, y, lw=2, c='k')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    kk = 0
    a = []
    while kk < len(hue):
        if hue[kk] < hue_thresh:
            b = kk
            kk += 1
            while kk < len(hue):
                if hue[kk] > hue_thresh:
                    break
                else:
                    kk += 1
            a.append([b, kk - 1])
        else:
            kk += 1

    st = (x[1] - x[0]) / 2.0
    for p in a:
        axes.axvspan(x[p[0]]-st, x[p[1]]+st, facecolor='g', alpha=0.5)
    plt.legend(legend_str)
    sns.despine()

    return fig, axes


##################################
#Preprocessing stuff from DeepEEG#
##################################

def mastoidReref(raw):
  ref_idx = pick_channels(raw.info['ch_names'],['M2'])
  eeg_idx = pick_types(raw.info,eeg=True)
  raw._data[eeg_idx,:] =  raw._data[eeg_idx,:]  -  raw._data[ref_idx,:] * .5 ;
  return raw

def GrattonEmcpRaw(raw):
  raw_eeg = raw.copy().pick_types(eeg=True)[:][0]
  raw_eog = raw.copy().pick_types(eog=True)[:][0]
  b = np.linalg.solve(np.dot(raw_eog,raw_eog.T), np.dot(raw_eog,raw_eeg.T))
  eeg_corrected = (raw_eeg.T - np.dot(raw_eog.T,b)).T
  raw_new = raw.copy()
  raw_new._data[pick_types(raw.info,eeg=True),:] = eeg_corrected
  return raw_new


def GrattonEmcpEpochs(epochs):
  '''
  # Correct EEG data for EOG artifacts with regression
  # INPUT - MNE epochs object (with eeg and eog channels)
  # OUTPUT - MNE epochs object (with eeg corrected)
  # After: Gratton,Coles,Donchin, 1983
  # -compute the ERP in each condition
  # -subtract ERP from each trial
  # -subtract baseline (mean over all epoch)
  # -predict eye channel remainder from eeg remainder
  # -use coefficients to subtract eog from eeg
  '''

  event_names = ['A_error','B_error']
  i = 0
  for key, value in sorted(epochs.event_id.items(), key=lambda x: (x[1], x[0])):
    event_names[i] = key
    i += 1

  #select the correct channels and data
  eeg_chans = pick_types(epochs.info, eeg=True, eog=False)
  eog_chans = pick_types(epochs.info, eeg=False, eog=True)
  original_data = epochs._data

  #subtract the average over trials from each trial
  rem = {}
  for event in event_names:
    data = epochs[event]._data
    avg = np.mean(epochs[event]._data,axis=0)
    rem[event] = data-avg

  #concatenate trials together of different types
  ## then put them all back together in X (regression on all at once)
  allrem = np.concatenate([rem[event] for event in event_names])

  #separate eog and eeg
  X = allrem[:,eeg_chans,:]
  Y = allrem[:,eog_chans,:]

  #subtract mean over time from every trial/channel
  X = (X.T - np.mean(X,2).T).T
  Y = (Y.T - np.mean(Y,2).T).T

  #move electrodes first
  X = np.moveaxis(X,0,1)
  Y = np.moveaxis(Y,0,1)

  #make 2d and compute regression
  X = np.reshape(X,(X.shape[0],np.prod(X.shape[1:])))
  Y = np.reshape(Y,(Y.shape[0],np.prod(Y.shape[1:])))
  b = np.linalg.solve(np.dot(Y,Y.T), np.dot(Y,X.T))

  #get original data and electrodes first for matrix math
  raw_eeg = np.moveaxis(original_data[:,eeg_chans,:],0,1)
  raw_eog = np.moveaxis(original_data[:,eog_chans,:],0,1)

  #subtract weighted eye channels from eeg channels
  eeg_corrected = (raw_eeg.T - np.dot(raw_eog.T,b)).T

  #move back to match epochs
  eeg_corrected = np.moveaxis(eeg_corrected,0,1)

  #copy original epochs and replace with corrected data
  epochs_new = epochs.copy()
  epochs_new._data[:,eeg_chans,:] = eeg_corrected

  return epochs_new


def PreProcess(raw, event_id, plot_psd=False, filter_data=True,
               filter_range=(1,30), plot_events=False, epoch_time=(-.2,1),
               baseline=(-.2,0), rej_thresh_uV=200, rereference=False, 
               emcp_raw=False, emcp_epochs=False, epoch_decim=1, plot_electrodes=False,
               plot_erp=False):

  sfreq = raw.info['sfreq']
  #create new output freq for after epoch or wavelet decim
  nsfreq = sfreq/epoch_decim
  tmin=epoch_time[0]
  tmax=epoch_time[1]
  if filter_range[1] > nsfreq:
    filter_range[1] = nsfreq/2.5  #lower than 2 to avoid aliasing from decim??

  #pull event names in order of trigger number
  event_names = ['A_error','B_error']
  i = 0
  for key, value in sorted(event_id.items(), key=lambda x: (x[1], x[0])):
    event_names[i] = key
    i += 1

  #Filtering
  if rereference:
    print('Rerefering to average mastoid')
    raw = mastoidReref(raw)

  if filter_data:
    print('Filtering Data Between ' + str(filter_range[0]) + 
            ' and ' + str(filter_range[1]) + ' Hz.')
    raw.filter(filter_range[0],filter_range[1],
               method='iir', verbose='WARNING' )

  if plot_psd:
    raw.plot_psd(fmin=filter_range[0], fmax=nsfreq/2 )

  #Eye Correction
  if emcp_raw:
    print('Raw Eye Movement Correction')
    raw = GrattonEmcpRaw(raw)

  #Epoching
  events = find_events(raw,shortest_event=1)
  color = {1: 'red', 2: 'black'}
  #artifact rejection
  rej_thresh = rej_thresh_uV*1e-6

  #plot event timing
  if plot_events:
    viz.plot_events(events, sfreq, raw.first_samp, color=color,
                        event_id=event_id)

  #Construct events - Main function from MNE
  epochs = Epochs(raw, events=events, event_id=event_id,
                  tmin=tmin, tmax=tmax, baseline=baseline,
                  preload=True,reject={'eeg':rej_thresh},
                  verbose=False, decim=epoch_decim)
  print('Remaining Trials: ' + str(len(epochs)))

  #Gratton eye movement correction procedure on epochs
  if emcp_epochs:
    print('Epochs Eye Movement Correct')
    epochs = GrattonEmcpEpochs(epochs)

  ## plot ERP at each electrode
  evoked_dict = {event_names[0]:epochs[event_names[0]].average(),
                              event_names[1]:epochs[event_names[1]].average()}

  # butterfly plot
  if plot_electrodes:
    picks = pick_types(evoked_dict[event_names[0]].info, meg=False, eeg=True, eog=False)
    fig_zero = evoked_dict[event_names[0]].plot(spatial_colors=True,picks=picks)
    fig_zero = evoked_dict[event_names[1]].plot(spatial_colors=True,picks=picks)

  # plot ERP in each condition on same plot
  if plot_erp:
    #find the electrode most miximal on the head (highest in z)
    picks = np.argmax([evoked_dict[event_names[0]].info['chs'][i]['loc'][2] 
              for i in range(len(evoked_dict[event_names[0]].info['chs']))])
    colors = {event_names[0]:"Red",event_names[1]:"Blue"}
    viz.plot_compare_evokeds(evoked_dict,colors=colors,
                            picks=picks,split_legend=True)

  return epochs


###################################
#CreateModel function from DeepEEG#
###################################

def CreateModel(feats,units=[16,8,4,8,16], dropout=.25,
                batch_norm=True, filt_size=3, pool_size=2):

  print('Creating ' +  feats.model_type + ' Model')
  print('Input shape: ' + str(feats.input_shape))


  nunits = len(units)

  ##---LSTM - Many to two, sequence of time to classes
  #Units must be at least two
  if feats.model_type == 'LSTM':
    if nunits < 2:
      print('Warning: Need at least two layers for LSTM')

    model = Sequential()
    model.add(LSTM(input_shape=(None, feats.input_shape[1]),
                   units=units[0], return_sequences=True))
    if batch_norm:
      model.add(BatchNormalization())
    model.add(Activation('relu'))
    if dropout:
      model.add(Dropout(dropout))

    if len(units) > 2:
      for unit in units[1:-1]:
        model.add(LSTM(units=unit,return_sequences=True))
        if batch_norm:
          model.add(BatchNormalization())
        model.add(Activation('relu'))
        if dropout:
          model.add(Dropout(dropout))

    model.add(LSTM(units=units[-1],return_sequences=False))
    if batch_norm:
      model.add(BatchNormalization())
    model.add(Activation('relu'))
    if dropout:
      model.add(Dropout(dropout))

    model.add(Dense(units=feats.num_classes))
    model.add(Activation("softmax"))


  ##---DenseFeedforward Network
  #Makes a hidden layer for each item in units
  if feats.model_type == 'NN':
    model = Sequential()
    model.add(Flatten(input_shape=feats.input_shape))

    for unit in units:
      model.add(Dense(unit))
      if batch_norm:
        model.add(BatchNormalization())
      model.add(Activation('relu'))
      if dropout:
        model.add(Dropout(dropout))

    model.add(Dense(feats.num_classes, activation='softmax'))

  ##----Convolutional Network
  if feats.model_type == 'CNN':
    if nunits < 2:
      print('Warning: Need at least two layers for CNN')
    model = Sequential()
    model.add(Conv2D(units[0], filt_size,
              input_shape=feats.input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))

    if nunits > 2:
      for unit in units[1:-1]:
        model.add(Conv2D(unit, filt_size, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size, padding='same'))


    model.add(Flatten())
    model.add(Dense(units[-1]))
    model.add(Activation('relu'))
    model.add(Dense(feats.num_classes))
    model.add(Activation('softmax'))

  ##----Convolutional Network
  if feats.model_type == 'CNN3D':
    if nunits < 2:
      print('Warning: Need at least two layers for CNN')
    model = Sequential()
    model.add(Conv3D(units[0], filt_size,
                     input_shape=feats.input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=pool_size, padding='same'))

    if nunits > 2:
      for unit in units[1:-1]:
        model.add(Conv3D(unit, filt_size, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=pool_size, padding='same'))


    model.add(Flatten())
    model.add(Dense(units[-1]))
    model.add(Activation('relu'))
    model.add(Dense(feats.num_classes))
    model.add(Activation('softmax'))


  ## Autoencoder
  #takes the first item in units for hidden layer size
  if feats.model_type == 'AUTO':
    encoding_dim = units[0]
    input_data = Input(shape=(feats.input_shape[0],))
    #,activity_regularizer=regularizers.l1(10e-5)
    encoded = Dense(encoding_dim, activation='relu')(input_data)
    decoded = Dense(feats.input_shape[0], activation='sigmoid')(encoded)
    model = Model(input_data, decoded)

    encoder = Model(input_data,encoded)
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = model.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))


  #takes an odd number of layers > 1
  #e.g. units = [64,32,16,32,64]
  if feats.model_type == 'AUTODeep':
    if nunits % 2 == 0:
      print('Warning: Please enter odd number of layers into units')

    half = nunits/2
    midi = int(np.floor(half))

    input_data = Input(shape=(feats.input_shape[0],))
    encoded = Dense(units[0], activation='relu')(input_data)

    #encoder decreases
    if nunits >= 3:
        for unit in units[1:midi]:
          encoded = Dense(unit, activation='relu')(encoded)

    #latent space
    decoded = Dense(units[midi], activation='relu')(encoded)

    #decoder increses
    if nunits >= 3:
      for unit in units[midi+1:-1]:
        decoded = Dense(unit, activation='relu')(decoded)

    decoded = Dense(units[-1], activation='relu')(decoded)

    decoded = Dense(feats.input_shape[0], activation='sigmoid')(decoded)
    model = Model(input_data, decoded)

    encoder = Model(input_data,encoded)
    encoded_input = Input(shape=(units[midi],))





  if feats.model_type == 'AUTO' or feats.model_type == 'AUTODeep':
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=opt, loss='mean_squared_error')



  if ((feats.model_type == 'CNN') or
      (feats.model_type == 'CNN3D') or
      (feats.model_type == 'LSTM') or
      (feats.model_type == 'NN')):

    # initiate adam optimizer
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
    # Let's train the model using RMSprop
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    encoder = []


  model.summary()

  return model, encoder


def TrainTestVal(model, feats, batch_size=2, 
                train_epochs=20, show_plots=True):

  print('Training Model:')
  # Train Model
  if feats.model_type == 'AUTO' or feats.model_type == 'AUTODeep':
    print('Training autoencoder:')

    history = model.fit(feats.x_train, feats.x_train,
                        batch_size = batch_size,
                        epochs=train_epochs,
                        validation_data=(feats.x_val,feats.x_val),
                        shuffle=True,
                        verbose=True,
                        class_weight=feats.class_weights
                       )

    # list all data in history
    print(history.history.keys())

    if show_plots:
      # summarize history for loss
      plt.semilogy(history.history['loss'])
      plt.semilogy(history.history['val_loss'])
      plt.title('model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'val'], loc='upper left')
      plt.show()

  else:
    history = model.fit(feats.x_train, feats.y_train,
              batch_size=batch_size,
              epochs=train_epochs,
              validation_data=(feats.x_val, feats.y_val),
              shuffle=True,
              verbose=True,
              class_weight=feats.class_weights
              )

    # list all data in history
    print(history.history.keys())

    if show_plots:
      # summarize history for accuracy
      plt.plot(history.history['acc'])
      plt.plot(history.history['val_acc'])
      plt.title('model accuracy')
      plt.ylabel('accuracy')
      plt.xlabel('epoch')
      plt.legend(['train', 'val'], loc='upper left')
      plt.show()
      # summarize history for loss
      plt.semilogy(history.history['loss'])
      plt.semilogy(history.history['val_loss'])
      plt.title('model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'val'], loc='upper left')
      plt.show()


    # Test on left out Test data
    score, acc = model.evaluate(feats.x_test, feats.y_test,
                                batch_size=batch_size)
    print(model.metrics_names)
    print('Test loss:', score)
    print('Test accuracy:', acc)

    # Build a dictionary of data to return
    data = {}
    data['score'] = score
    data['acc'] = acc

    return model, data