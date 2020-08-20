# Source: https://dsp.stackexchange.com/questions/45345/how-to-correctly-compute-the-eeg-frequency-bands-with-python

# Art screen imports
from psychopy import visual

# EEG imports
import math
import mne
import numpy as np
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, lfilter, lfilter_zi, firwin
from time import sleep

buffer = 250 # sampling rate
channels = 16



class CircularBuffer:
    def __init__(self, chunks):
        # empty array of channels * (chunks*s) for baseline
        self.window = np.zeros((channels, buffer*chunks))
        self.chunks = chunks
        self.window_read = self.chunks // 2
        self.window_write = 0
        self.chunk_size = buffer

    def read(self):
        return self.window

    def write(self, data):
        # fill in window with one chunk of data
        chunk_start = self.window_write * self.chunk_size
        chunk_end = (self.window_write + 1) * self.chunk_size

        self.window[:, chunk_start:chunk_end] = data
        self.window_write = (self.window_write + 1) % self.chunks
        self.window_read = (self.window_read + 1) % self.chunks


class Stream:
    def __init__(self):
        Thread.__init__(self)

        self.buffer = buffer
        print("looking for an EEG stream...")
        self.streams = resolve_byprop('type', 'EEG', timeout=2)

        if len(self.streams) == 0:
            raise(RuntimeError("Cant find EEG stream"))
        print("Start aquiring data")

        self.stream = self.streams[0]

        self.inlet = StreamInlet(self.stream, max_chunklen-buffer)
        self.count = 0
        self.chunks = 5
        self.avg_len = 10
        self.lazy_low = 0.9
        self.lazy_high = 1.1


        self.buf = CircularBuffer(self.chunks)

        self.noise = {
                    'Delta': False,
                    'Theta': False,
                    'Alpha': False,
                    'Beta': False
                    }

        self.state = {
                    'Delta': 'Low',
                    'Theta': 'Low',
                    'Alpha': 'Low',
                    'Beta': 'Low'
                    }

        self.low_bound = {
                        'Delta': self.lazy_low,
                        'Theta': self.lazy_low,
                        'Alpha': self.lazy_low,
                        'Beta': self.lazy_low
                        }

        self.high_bound = {
                        'Delta': self.lazy_high,
                        'Theta': self.lazy_high,
                        'Alpha': self.lazy_high,
                        'Beta': self.lazy_high
                        }

        self._stop_loop = False


    def stop(self):
        self._stop_loop = True


    def run(self):
        eeg_bands = {'Delta': (0, 4),
                     'Theta': (4, 7),
                     'Alpha': (8, 15),
                     'Beta': (16, 31)}

        # will store average values for each band
        self.avg = dict()
        for band in eeg_bands:
            self.avg[band] = 0

        # 0.9 if avg_len=10, means that a value's relative contribution to the avg decays to 1/e after 10 iterations
        avg_param = 1 - (1 / self.avg_len)

        while True:
            # Sample is a 2d array of [ [channel_i]*channels ] * buffer
            samples, timestamps = self.inlet.pull_chunk(timeout=2.0, max_samples=self.buffer)
            if timestamps:
                data = np.vstack(samples)
                data = np.transpose(data)
                print(np.shape(data))
                self.buf.write(data)

            # Check that the buffer is filled before creating baseline
            if self.count >= self.chunks:
                current_data = self.buf.window

                # converts data from time domain to frequency domain
                fft_data = np.absolute(np.fft.rfft(current_data))
                fft_freqs = np.fft.rfftfreq(len(current_data.T), 1.0/buffer)

                bias_correction = 1 - (avg_param**(self.count-self.chunks+1)) # prevents skew in avg during early iterations

                # for each band, identify measurements corresponding to appropriate frequency, average across channels
                for band in eeg_bands:
                    band_idx = np.where((fft_freqs >= eeg_bands[band][0]) and (fft_freqs <= eeg_bands[band][1]))[0]
                    freq_val = np.mean(fft_data[band_idx])

                    if (freq_val > self.avg[band] * self.low_bound[band]) and (freq_val < self.avg[band] * self.high_bound[band]):
                        self.noise[band] = False
                        if freq_val < self.avg[band]:
                            self.state[band] = 'Low'
                        elif freq_val > self.avg[band]:
                            self.state[band] = 'High'
                    else:
                        self.noise[band] = True

                    # calculate exponentially weighted average for a given band, store in avg
                    if not self.noise[band]:
                        self.avg[band] = ((avg_param * avg[band]) + ((1 - avg_param) * freq_val)) / bias_correction

                    print(band, 'Amplitude: ', freq_val)

                print('State: ', self.state)
                print('Noise: ', self.noise)
                print('Average: ', self.avg)

if __name__ == '__main__':
    t = Stream()
    t.start()
