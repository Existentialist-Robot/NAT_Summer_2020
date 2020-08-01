# Source: https://dsp.stackexchange.com/questions/45345/how-to-correctly-compute-the-eeg-frequency-bands-with-python

# Art screen imports
from psychopy import visual

#!/usr/bin/env python
# EEG imports
import math
import mne
import numpy as np
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, lfilter, lfilter_zi, firwin
from threading import Thread
from time import sleep

buffer = 250 # sampling rate
channels = 16

# Querying open streams matching this name
print("looking for an EEG stream...")
streams = resolve_byprop('type', 'EEG', timeout=2)

if len(streams) == 0:
    raise(RuntimeError("Cant find EEG stream"))
print("Start aquiring data")

stream = streams[0]

# initializing stream, max chunk 1 s
inlet = StreamInlet(stream, max_chunklen=buffer)

# creates a window to draw art on-screen
class artScreen:
    def __init__(self,inputSize):
        self.size = inputSize
        self.window = visual.Window(size=self.size, pos=[960,540], fullscr=False, allowGUI=True, monitor="testMonitor")
        self.artImage = visual.ImageStim(self.window)
        self.artImage.draw(self.window)
        self.window.flip()

    def updateScreen(self,newImage):
        """update the art screen"""
        self.artImage.image = newImage
        self.artImage.draw(self.window)
        self.window.flip()


class CircularBuffer:
    def __init__(self, chunks):
        # empty array of channels * (chunks*s) for baseline
        self.window = np.zeros((channels, buffer*chunks))
        self.chunks = chunks
        self.window_read = self.chunks // 2
        self.window_write = 0
        self.chunk_size = buffer

    def read(self):
        #return self.window[:][self.window_read*self.chunk_size:(self.window_read+1)*self.chunk_size]
        return self.window

    def write(self, data):
        # fill in window with one chunk of data
        chunk_start = self.window_write * self.chunk_size
        chunk_end = (self.window_write + 1) * self.chunk_size

        self.window[:, chunk_start:chunk_end] = data
        self.window_write = (self.window_write + 1) % self.chunks
        self.window_read = (self.window_read + 1) % self.chunks


class Stream (Thread):
    def __init__(self):
        Thread.__init__(self)

        self.BUFFER = buffer
        print("looking for an EEG stream...")
        self.streams = resolve_byprop('type', 'EEG', timeout=2)

        if len(self.streams) == 0:
            raise(RuntimeError("Cant find EEG stream"))
        print("Start aquiring data")

        self.stream = self.streams[0]

        self.inlet = StreamInlet(self.stream, max_chunklen=BUFFER)
        self.count = 0
        self.chunks = 5
        self.avg_len = 10

        self.buf = CircularBuffer(self.chunks)
        self.state = 'noise'
        self._stop_loop = False


    def stop(self):
        self._stop_loop = True


    def run(self):
        eeg_bands = {'Delta': (0, 4),
                     'Theta': (4, 7),
                     'Alpha': (8, 15),
                     'Beta': (16, 31)}

        # dict of lists, each will store the 10 most recent recordings for a
        # given band
        eeg_bands_fft = dict()
        for band in eeg_bands:
            eeg_bands_fft[band] = []

        # dict to store average across lists in eeg_bands_fft
        avg = dict()

        while True:
            # Sample is a 2d array of [ [channel_i]*channels ] * buffer
            samples, timestamps = self.inlet.pull_chunk(timeout=2.0, max_samples=self.BUFFER)
            if timestamps:
                data = np.vstack(samples)
                data = np.transpose(data)
                print(np.shape(data))
                self.buf.write(data)

            # Check that the buffer is filled before creating baseline
            if self.count >= self.chunks
                current_data = self.buf.window

                # convert data from time domain to frequency domain
                fft_data = np.absolute(np.fft.rfft(current_data))
                fft_freqs = np.fft.rfftfreq(len(current_data.T), 1.0/buffer)

                # for each band, identify corresponding measurements from frequency domain,
                # get average amplitude across each of these points
                for band in eeg_bands:
                    band_idx = np.where((fft_freq >= eeg_bands[band][0]) & (fft_freq <= eeg_bands[band][1]))[0]
                    freq_val = np.mean(fft_data[band_idx])

                    # running baseline, if there are more than avg_len recordings in
                    # eeg_bands_fft, drop the first one
                    if self.count >= self.avg_len:
                        eeg_bands_fft[band].pop(0)

                    # append the current average to the list. Stored in order, and we
                    # can weigh these values by receny
                    eeg_bands_fft[band].append(freq_val)

                    # stores the current baseline's average in avg
                    avg[band] = sum(eeg_bands_fft[band]) / len(eeg_bands_fft[band])


if __name__ == '__main__':
    t = Stream()
    t.start()
