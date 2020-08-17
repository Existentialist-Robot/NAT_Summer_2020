"""Example program to demonstrate how to send a multi-channel time series to
LSL."""

import random
import time
import numpy as np

from pylsl import StreamInfo, StreamOutlet

def sendingData():
    # first create a new stream info (here we set the name to BioSemi,
    # the content-type to EEG, 8 channels, 100 Hz, and float-valued data) The
    # last value would be the serial number of the device or some other more or
    # less locally unique identifier for the stream as far as available (you
    # could also omit it but interrupted connections wouldn't auto-recover).
    info = StreamInfo('BioSemi', 'EEG', 16, 250, 'float32', 'myuid34234')

    # next make an outlet
    outlet = StreamOutlet(info)

    print("now sending data...")
    
    alpha_low_bound = 8
    beta_low_bound = 16
    delta_low_bound = 0
    theta_low_bound = 4

    # y = np.sin(np.ones((250,)) * (x * (alpha_low_bound+alpha_random)))*8

    x = np.linspace(0, 2*np.pi, 250)

        # eeg_bands = {'Delta': (0, 4),
        #              'Theta': (4, 7),
        #              'Alpha': (8, 15),
        #              'Beta': (16, 31)}

    baseline_alpha = np.sin(np.ones((250,)) * (x * (alpha_low_bound+3))) * 4
    baseline_beta = np.sin(np.ones((250,)) * (x * (beta_low_bound+8))) * 4
    baseline_delta = np.sin(np.ones((250,)) * (x * (delta_low_bound+2))) * 4
    baseline_theta  = np.sin(np.ones((250,)) * (x * (theta_low_bound+1.5))) * 4

    baseline = np.sum(baseline_alpha, baseline_beta, baseline_delta, baseline_theta)


    real_fake_alpha = np.sin(np.ones((250,)) * (x * (alpha_low_bound+3))) * 6 
    real_fake_beta = np.sin(np.ones((250,)) * (x * (beta_low_bound+8))) * 2
    real_fake_delta = np.sin(np.ones((250,)) * (x * (delta_low_bound+2))) * 18
    real_fake_theta = np.sin(np.ones((250,)) * (x * (theta_low_bound+1.5))) * 0
    
    real_fake_data = np.sum(real_fake_alpha,real_fake_beta,real_fake_delta,real_fake_theta)


# Psuedo-randomized frequency
    # alpha_random = random.random() * 7
    # beta_random = random.random() * 10
    # delta_random = random.random() * 4
    # theta_random = random.random() * 3
    
    # alpha = np.sin(np.ones((250,)) * np.pi/(alpha_low_bound+alpha_random)
    # beta = np.sin(np.ones((250,)) * np.pi/(beta_low_bound+beta_random))
    # delta = np.sin(np.ones((250,)) * np.pi/(delta_low_bound+delta_random))
    # theta  = np.sin(np.ones((250,)) * np.pi/(theta_low_bound+theta_random))


    count = 0
    while True:
        # make a new random 8-channel sample; this is converted into a
        # pylsl.vectorf (the data type that is expected by push_sample)
        if count < 2500:
            baseline_current = baseline[count % 250]
            mysample = [baseline_current, baseline_current, baseline_current,
                        baseline_current, baseline_current, baseline_current,
                        baseline_current, baseline_current, baseline_current,
                        baseline_current, baseline_current, baseline_current,
                        baseline_current, baseline_current, baseline_current,
                        baseline_current]
        else:
            real_fake_data_current = real_fake_data[count % 250]
            mysample = [real_fake_data_current, real_fake_data_current, real_fake_data_current,
                        real_fake_data_current, real_fake_data_current, real_fake_data_current,
                        real_fake_data_current, real_fake_data_current, real_fake_data_current,
                        real_fake_data_current, real_fake_data_current, real_fake_data_current,
                        real_fake_data_current, real_fake_data_current, real_fake_data_current,
                        real_fake_data_current]

        # now send it and wait for a bit
        outlet.push_sample(mysample)
        time.sleep(0.004)
        count += 1
        #print("sending")
