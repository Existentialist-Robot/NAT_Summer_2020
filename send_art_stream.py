"""Example program to demonstrate how to send a multi-channel time series to
LSL."""

import random
import time

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

    alpha_random = random.random() * 7
    beta_random = random.random() * 10
    delta_random = random.random() * 4
    theta_random = random.random() * 3

    alpha = np.sin(np.ones((250,)) * np.pi/(alpha_low_bound+alpha_random)
    beta = np.sin(np.ones((250,)) * np.pi/(beta_low_bound+beta_random))
    delta = np.sin(np.ones((250,)) * np.pi/(delta_low_bound+delta_random))
    theta  = np.sin(np.ones((250,)) * np.pi/(theta_low_bound+theta_random))


    while True:
        # make a new random 8-channel sample; this is converted into a
        # pylsl.vectorf (the data type that is expected by push_sample)
        mysample = [random.random(), random.random(), random.random(),
                    random.random(), random.random(), random.random(),
                    random.random(), random.random(), random.random(), 
                    random.random(), random.random(), random.random(), 
                    random.random(), random.random(), random.random(), 
                    random.random()]
        # now send it and wait for a bit
        outlet.push_sample(mysample)
        time.sleep(0.004)
        #print("sending")
