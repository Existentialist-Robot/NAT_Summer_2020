"""Example program to demonstrate how to send a multi-channel time series to
LSL."""

# import random
# import time

# from pylsl import StreamInfo, StreamOutlet

from pyOpenBCI import OpenBCICyton
from pylsl import StreamInfo, StreamOutlet
import numpy as np

SCALE_FACTOR_EEG = (4500000)/24/(2**23-1) #uV/count
SCALE_FACTOR_AUX = 0.002 / (2**4)

def sendingData():
    # first create a new stream info (here we set the name to BioSemi,
    # the content-type to EEG, 8 channels, 100 Hz, and float-valued data) The
    # last value would be the serial number of the device or some other more or
    # less locally unique identifier for the stream as far as available (you
    # could also omit it but interrupted connections wouldn't auto-recover).


    # info = StreamInfo('BioSemi', 'EEG', 8, 100, 'float32', 'myuid34234')

    # # next make an outlet
    # outlet = StreamOutlet(info)

    # print("now sending data...")
    
    # while True:
    #     # make a new random 8-channel sample; this is converted into a
    #     # pylsl.vectorf (the data type that is expected by push_sample)
    #     mysample = [random.random(), random.random(), random.random(),
    #                 random.random(), random.random(), random.random(),
    #                 random.random(), random.random()]
    #     # now send it and wait for a bit
    #     outlet.push_sample(mysample)
    #     time.sleep(0.004)    
    # print(address)

    # print(name)
    # print("Creating LSL stream for EEG. \nName: OpenBCIEEG\nID: OpenBCItestEEG\n")

    info_eeg = StreamInfo('OpenBCIEEG', 'EEG', 16, 250, 'float32', 'OpenBCItestEEG')

    # # print("Creating LSL stream for AUX. \nName: OpenBCIAUX\nID: OpenBCItestEEG\n")

    # # info_aux = StreamInfo('OpenBCIAUX', 'AUX', 3, 250, 'float32', 'OpenBCItestAUX')

    info_eeg.desc().append_child_value("manufacturer", "OpenBCI")
    eeg_channels = info_eeg.desc().append_child("channels")

    for c in ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2', 'F7', 'F8', 'F3', 'F4', 'T7', 'T8', 'P3', 'P4']:
        eeg_channels.append_child("channel") \
            .append_child_value("label", c) \
            .append_child_value("unit", "microvolts") \
            .append_child_value("type", "EEG")

    # # eeg_outlet = StreamOutlet(eeg_info, LSL_EEG_CHUNK)

    outlet_eeg = StreamOutlet(info_eeg)
    # # outlet_aux = StreamOutlet(info_aux)

    def lsl_streamers(sample):
        outlet_eeg.push_sample(np.array(sample.channels_data)*SCALE_FACTOR_EEG)
        # outlet_aux.push_sample(np.array(sample.aux_data)*SCALE_FACTOR_AUX)

    board = OpenBCICyton(port='COM3', daisy=True)

    board.start_stream(lsl_streamers)
    #     #print("sending")


