class frequency:
    def __init__(self):
        self.signals = {
            'a':[],
            'b':[],
            't':[],
            'd':[],
        }

    def store_signal(self,alpha,beta,theta,delta):

        '''store the 4 brain wave signals (and return the last index of the signals)'''

        self.signals['a'].append(alpha)
        self.signals['b'].append(beta)
        self.signals['t'].append(theta)
        self.signals['d'].append(delta)
        # return len(self.signals['a']) - 1
        # return self.get_final_index()

    # def get_len(self):
    #     ''' return the length of the signals '''
    #     return len(self.signals['a'])

    def get_final_index(self):

        ''' return the last index of the signals '''

        return (len(self.signals['a']) - 1)
        
    def get_item(self,position,type = 'all'):

        ''' get values in a specific (or just last??) position in all or specific signals in the freq
        Default is all signals
        '''

        last = self.get_final_index()
        # ???
        if position > last:
            position = last
        return_value = self.signals.get(type)
        if not return_value:    # if wanting to get all signals
            return tuple(item[position] for item in self.signals.values())  # iterate through each signal and grab the value at the given position (or just last?) and store it in a tuple
        return return_value[position]   # return the value in the specific position in the specific signal

    def binarize(self,type = 'all'):

        ''' binarize the current values of all of or specific types of signals
        Default is all types of signals
        A value is binarized to 1 if it's above the mean of the corresponding signal, 0 if it's smaller
        '''
        
        # get the current values from signals of specified type or all by default
        item = self.get_item(self.get_final_index(),type)
        # get the current averages of each signal
        means = self.mean(type)
        # iterate through the current values of each signal and the corresponding average of the signals. if the current value is smaller than the average, store 0 in the tuple. if the current value is larger than the average, store 1. return the tuple
        return tuple(0 if num <= mean else 1 for num,mean in zip(item,means))

    def get_lists(self,type = 'all'):

        ''' get the whole list of values for signals of specified or all types
        Default is all types
        '''

        return_value = self.signals.get(type)
        if not return_value:    # if wanting all types, return values of all signals
            return tuple(self.signals.values())
        return return_value

    def mean(self,type = 'all'):

        ''' return the average of all of or specific types of signals
        Default is all types
        '''

        if self.get_final_index() == 0: # if there is only one value for each signal, return some pre-determined thresholds
            # TODO: Ask WHERE DID HE GET THESE THRESHOLDS?
            if type == 'a':
                return 10,
            elif type == 'b':
                return 8.75,
            elif type == 't':
                return 6,
            elif type == 'd':
                return 1.5,
            else:
                return (10,8.75,6,1.5)
        lists = self.get_lists(type)
        return tuple(sum(item)/len(item) for item in lists) # iterate through each list of values for each signal, and store its mean in the return tuple

    def stdev(self,type = 'all'):

        ''' return the standard deviation of all or specific types of signal
        Default is all types
        '''

        if self.get_final_index() == 0:
            return 0
        else:
            lists = self.get_lists(type)
            return tuple(np.std(item) for item in lists)
     



        


#switch around the bits right before
#cap randomize
#combine and store warped bit planes
