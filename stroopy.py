import numpy as np
from pandas import DataFrame
from psychopy import visual, core, event
from time import time, strftime, gmtime
from optparse import OptionParser
from pylsl import StreamInfo, StreamOutlet
from glob import glob
from random import choice
from math import floor


def present(duration=120):

    # create 
    info = StreamInfo('Markers', 'Markers', 1, 0, 'int32', 'myuidw43536')

    # next make an outlet
    outlet = StreamOutlet(info)

    markernames = [1, 2]

    start = time()

    n_trials = floor(duration / 3.5)
    iti = 1
    soa = 2.5
    jitter = 0.3
    record_duration = np.float32(duration)

    # Setup log
    position = np.random.binomial(1, 0.15, n_trials)

    trials = DataFrame(dict(position=position,
                            timestamp=np.zeros(n_trials)))

    # graphics

    def loadImage(filename):
        return visual.ImageStim(win=mywin, image=filename)

    mywin = visual.Window([1920, 1080], monitor="testMonitor", units="deg",
                          fullscr=True)
    targets = list(
        map(loadImage, glob('stimulus_presentation/faces/happy/*.jpg')))
    nontargets = list(
        map(loadImage, glob('stimulus_presentation/faces/sad/*.jpg')))

    for ii, trial in trials.iterrows():
        # inter trial interval
        core.wait(iti + np.random.rand() * jitter)

        # onset
        pos = trials['position'].iloc[ii]
        image = choice(targets if pos == 1 else nontargets)
        image.draw()
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
