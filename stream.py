from pyOpenBCI import OpenBCICyton
from pylsl import StreamInfo, StreamOutlet
import numpy as np

SCALE_FACTOR_EEG = (4500000)/24/(2**23-1) #uV/count
SCALE_FACTOR_AUX = 0.002 / (2**4)

def stream(address,name):
    print(address)
    print(name)
    print("Creating LSL stream for EEG. \nName: OpenBCIEEG\nID: OpenBCItestEEG\n")

    info_eeg = StreamInfo('OpenBCIEEG', 'EEG', 16, 250, 'float32', 'OpenBCItestEEG')

    # print("Creating LSL stream for AUX. \nName: OpenBCIAUX\nID: OpenBCItestEEG\n")

    # info_aux = StreamInfo('OpenBCIAUX', 'AUX', 3, 250, 'float32', 'OpenBCItestAUX')

    info_eeg.desc().append_child_value("manufacturer", "OpenBCI")
    eeg_channels = info_eeg.desc().append_child("channels")

    for c in ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2', 'F7', 'F8', 'F3', 'F4', 'T7', 'T8', 'P3', 'P4']:
        eeg_channels.append_child("channel") \
            .append_child_value("label", c) \
            .append_child_value("unit", "microvolts") \
            .append_child_value("type", "EEG")

    # eeg_outlet = StreamOutlet(eeg_info, LSL_EEG_CHUNK)


    outlet_eeg = StreamOutlet(info_eeg)
    # outlet_aux = StreamOutlet(info_aux)

    def lsl_streamers(sample):
        outlet_eeg.push_sample(np.array(sample.channels_data)*SCALE_FACTOR_EEG)
        # outlet_aux.push_sample(np.array(sample.aux_data)*SCALE_FACTOR_AUX)

    board = OpenBCICyton(port='COM3', daisy=True)

    board.start_stream(lsl_streamers)