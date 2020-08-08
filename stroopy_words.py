import numpy as np
from pandas import DataFrame
from psychopy import visual, core, event
from time import time, strftime, gmtime
from optparse import OptionParser
from pylsl import StreamInfo, StreamOutlet
from glob import glob
from random import choice
from math import floor
from csv import reader


def present(duration=120):

    # create
    info = StreamInfo('Markers', 'Markers', 1, 0, 'int32', 'myuidw43536')

    # next make an outlet
    outlet = StreamOutlet(info)

    markernames = [1, 2, 3]

    start = time()

    n_trials = floor(duration / 3.5)
    iti = 1
    soa = 2.5
    jitter = 0.3
    record_duration = np.float32(duration)

    # Setup log
    position = np.random.randint(3, size=n_trials)

    trials = DataFrame(dict(position=position,
                            timestamp=np.zeros(n_trials)))

    mywin = visual.Window([1920, 1080], monitor="testMonitor", units="deg",
                          fullscr=True)

    word_files = glob(r'stimulus_presentation\words\*.txt')
    negative = open(word_files[0])
    neutral = open(word_files[1])
    positive = open(word_files[2])

    negative_words = list(reader(negative, delimiter=' '))
    neutral_words = list(reader(neutral, delimiter=' '))
    positive_words = list(reader(positive, delimiter=' '))

    # converts from multilist to list
    negative_words = [item for negative_list in negative_words for item in negative_list]
    neutral_words = [item for neutral_list in neutral_words for item in neutral_list]
    positive_words = [item for positive_list in positive_words for item in positive_list]

    for ii, trial in trials.iterrows():
        # inter trial interval
        core.wait(iti + np.random.rand() * jitter)

        # onset
        pos = trials['position'].iloc[ii]

        if pos == 0:
            word = choice(negative_words)
        elif pos == 1:
            word = choice(neutral_words)
        else:
            word = choice(positive_words)

        text = visual.TextStim(win=mywin, text=word, units='pix', font='Arial', height=175, alignHoriz='center')
        text.draw()

        timestamp = time()
        outlet.push_sample([markernames[pos]], timestamp)
        mywin.flip()

        # offset
        core.wait(soa)
        mywin.flip()
        if len(event.getKeys()) > 0 or (time() - start) > record_duration:
            break
        event.clearEvents()
    # Cleanup
    mywin.close()


def main():
    parser = OptionParser()

    parser.add_option("-d", "--duration",
                      dest="duration", type='int', default=120,
                      help="duration of the recording in seconds.")

    (options, args) = parser.parse_args()
    present(options.duration)

if __name__ == '__main__':
    main()
