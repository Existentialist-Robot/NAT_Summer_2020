from image_util import Warps, Admin
import numpy as np
import random
from PIL import Image
from functools import reduce

class bin_image(Warps):
    def __init__(self,size,freq,order,tmp_mode = 1,warp = [None,None],dim = None):
        self.dim = dim
        self.warp = warp
        self.tmp_mode = tmp_mode    # template mode
        self.order = order  # the order by which the binarized versions of freq are binarized and stored (see self.binarized())
        self.freq = freq
        self.admin = Admin(self.freq,self.tmp_mode,self.warp)
        self.x,self.y = size
        self.number_bits = 0
        self.binary_image = np.zeros((self.x,self.y,3,8),np.uint8)  # 8 arrays of x by y by 3
        self.image = np.zeros((self.x,self.y,3),np.uint8)   # array of x by y by 3, for the final image?
        self.binary_reset_image = None
        self.last_plane = None
        self.divisons_x = 1 # how many subdivisions there are in the x dimension
        self.divisons_y = 1 # how many subdivisions there are in the y dimension
        self.current_division_x = 0
        self.current_division_y = 0
        self.create_tmp()

    def create_tmp(self):
        ''' create template
        Depending on the template mode (1,3,2,-1), set the binary_rest_image, last_plane, and/or x and y divisions
        '''
        if self.tmp_mode == 3:
            self.binary_reset_image = np.zeros((self.x,self.y,3,8),np.uint8)    # 8 arrays of x by y by 3
            self.last_plane = np.zeros((self.x,self.y,3),np.uint8)  # array of x by y by 3
        elif self.tmp_mode == 2 or self.tmp_mode == -1:
            # dims is a tuple with 2 numbers for subdividing the image?
            dims = self.dim
            if dims == None:
                dims = random.choice([(1,8),(8,1),(2,4),(4,2)]) # randomly choose the subdivisions to be 1x8, 8x1, 2x4, or 4x2
            self.divisons_x,self.divisons_y = dims

    def increment_count(self):

        ''' increment the current bit number and subdivision index by 1 '''

        self.number_bits += 1
        if self.tmp_mode == 2 or self.tmp_mode == -1:
            self.current_division_x += 1
            if self.current_division_x == self.divisons_x:
                self.current_division_x = 0
                self.current_division_y += 1
                self.current_division_y %= self.divisons_y
                
        return self.number_bits

    def convert(self,item):

        ''' converts the binary number into decimal form

        essentially does:

        result = 2*item[0] + item[1]
        for i in range(2,len(item)):
            result = 2*result + item[i+1]

        item is a 1x8 array, representing the binary value of the byte at a specific pixel '''

        return reduce(lambda x,y: 2*x + y,item)

    def save_images(self):
        if self.number_bits != 7 or self.warp[0] == None:
            image = Image.fromarray(self.image)
        name = self.admin.get_name(self.number_bits)
        if self.number_bits == 7:
            if self.tmp_mode == 3:
                plane = Image.fromarray(self.last_plane)
                plane.save(name.replace("final_image","8"))
            if self.warp[0] == 't':
                image = Image.fromarray(self.trig(self.image,self.warp[1]))
        image.save(name)
        return image

    def create_image(self):
        # binarize the current freq signal values in the given order, i.e. self.order
        binarized = self.binarize() # binarized is a tuple with binary numbers
        self.store_bin_image(binarized)
        self.store_image()
        image = self.save_images()
        count = self.increment_count()
        if count == 8:
            self.reset()
        return image

    def store_bin_image(self,bin):

        ''' set the RGB binary numbers for each subdivision, only for the current bit number
        bin is a list of binary numbers indicating which signals were above/below boundary
        '''

        increment_x = self.x//self.divisons_x   # how big each subdivision is in the x dimension
        increment_y = self.y//self.divisons_y   # how big each subdivision is in the y dimension
        x_end = (self.current_division_x + 1) * increment_x # the boundary of the current division in the x dimension
        y_end = (self.current_division_y + 1) * increment_y # the boundary of the current division in the y dimension
        
        # if template mode is -1 (will this yield just a flat-colour image?) set the x and y ends to the x and y image sizes
        if self.tmp_mode == -1:
            x_end = self.x
            y_end = self.y

        # iterate through the subdivisions
        for i in range(self.current_division_x * increment_x,x_end ):
            for j in range(self.current_division_y * increment_y, y_end):
                for k in range(3):  # iterate through RGB
                    self.binary_image[i][j][k][self.number_bits] = bin[k]   # this only uses the first 3 numbers of bin
                    if self.tmp_mode == 3:
                        self.binary_reset_image[i][j][k][self.number_bits] = bin[k]

    def store_image(self):
        # iterate through RGB for each pixel
        for i in range(self.x):
            for j in range(self.y):
                for k in range(3):
                    if  self.tmp_mode < 3 or self.number_bits == 7:
                        # set the RGB values of the current pixel 
                        self.image[i][j][k] = self.convert(self.binary_image[i][j][k])
                            
                        if self.tmp_mode == 3 and self.number_bits == 7:
                            self.last_plane[i][j][k] = self.convert(self.binary_reset_image[i][j][k])
                    else: 
                        self.image[i][j][k] = self.convert(self.binary_reset_image[i][j][k])
        
        self.reset_tmp()

    def binarize(self):
        binarized = self.freq.binarize()
        swapped = [0,0,0,0]
        for i in range(len(binarized)): # iterate through the binarization of the current values
            # swap the order of binarized in the order of self.order
            swapped[int(self.order[i])] = binarized[i]           
        return tuple(swapped)   # return the swapped binarized as a tuple

    def reset(self):

        ''' reset binary image, image, and current bit number '''

        self.binary_image = np.zeros((self.x,self.y,3,8),np.uint8)
        self.image = np.zeros((self.x,self.y,3),np.uint8)
        self.number_bits = 0

    def reset_tmp(self):
        if self.tmp_mode == 3:
            self.binary_reset_image = np.zeros((self.x,self.y,3,8),np.uint8)


