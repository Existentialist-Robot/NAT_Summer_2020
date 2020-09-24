from tensorflow.keras import models
import numpy as np
from mne.filter import filter_data


from utils.utils import *
import os



def make_model(model_path):
    '''
    hyper parameters to play around with
    filt_size: convolution size
    input dims -> (num_epochs, 625,16)
    '''


    #load and concatenate all csv's from data into raw 
    raw = load_data(sfreq = 250)

    #define event ID's
    event_id = {'negative':1, 'neutral':2, 'positive':3}

    #convert events from raw into epochs
    epochs = PreProcess(raw,event_id,rej_thresh_uV=300,epoch_time=(-.1,2.5), baseline=(-.1,0))

    # Create features for CNN classifier
    feats = FeatureEngineer(epochs,model_type='CNN',frequency_domain = False,
                            normalization=True)

    model, _ = CreateModel(feats)

    model,_ = TrainTestVal(model,feats,train_epochs=5)

    model(model.save(model_path))


class LiveModel:
    def __init__(self,model_path, model_q,art_q):

        self.last_chunk = None #the previous 250 chunk
        self.current_chunk = None #the newest 250 chunk
        self.model_q = model_q #queue to recieve from running stream
        self.art_q = art_q #queue to send to artScreen
        self.model = models.load_model(modelPath)

        #preprocessing parameters
        self.l_freq = l_freq
        self.h_freq = h_freq

    def getData(self):
        #grab new data from running_stream
        new_chunk = self.model_q.get() # shape is (16,250)

        if self.last_chunk == None:
            self.last_chunk = new_chunk #put this chunk in the first timestep
            self.current_chunk = self.model_q.get() #wait and get another chunk for the next timestep

        else:
            self.last_chunk = self.current_chunk #shift to first timestep
            self.current_chunk = new_chunk #put newest chunk in next timestep
        
        return np.concatenate(self.last_chunk,self.new_chunk) #shape (16,500)
    
    def preprocess(self,data):

        '''
        data in shape (16,500)
        '''
        #convert nanoVolts to Volts
        data *= 1e-9
        #filter data
        data = filter_data(data, method='iir',l_freq = self.l_freq, h_freq = self.h_freq)
        #reshape from (16,500) to (500,16)
        data = np.moveaxis(data,0,1)
        
        #expand 2nd dimention for convolutions (500,16) --> (500,16,1)
        data = np.expand_dims(data,2)
        #expand 1st dimention to input shape of CNN (500,16,1) --> (1,500,16,1)
        data = np.expand_dims(data,0)
        #normalize data
        data = (data - np.mean(data)) / np.std(data)
        #convert to float32
        data = data.astype(float32)

        return data
    
    def predictAndSend(self, data):
        data = self.getData() #retrieve data
        data = self.preprocess(data) #preprocess data

        prediction = self.model.predict(data) #make prediction [neg,neut,pos]

        self.art_q.put(prediction)
    
    def run(self):
        while True:
            self.predictAndSend()






        
c = CircularBuffer(5)
print(c.chunk_size)

