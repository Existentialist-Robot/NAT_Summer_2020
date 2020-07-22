from multiprocessing import Process
import record
import stroopy
import stream

def ButtonFuncStartStream_onPress():
    address = '123456789'
    name = 'EdenBCI'
    stream_process = Process(target=stream.stream, args=(address,name))
    stream_process.start()

def ButtonFuncStartBaseline_onPress():
    recording_path = os.path.join(os.path.expanduser("~"), "eeg-notebooks", "data", "visual", "Stroop", "subject" + "_" + str(subject), "session" + "_" + str(session), ("recording_%s.csv" %
                                              strftime("%Y-%m-%d-%H.%M.%S", gmtime())))
    print('Recording data to: ', recording_path)

    stimulus = Process(target=stroopy.present, args=(duration,))
    recording = Process(target=record.record, args=(duration, recording_path))

    stimulus.start()
    recording.start()
    print('finished')

def ButtonListener_for_this_one_button():
    ButtonFuncStartStream_onPress()
