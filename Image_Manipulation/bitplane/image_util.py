import os,random
import numpy as np

class Warps:
    def trig(self,image,forced):
        if not forced:
            func = random.choice([np.cos,np.sin,np.tan])
        elif forced.lower() == 'c':
            func = np.cos
        elif forced.lower() == 's':
            func == np.sin
        elif forced.lower() == 't':
            func = np.tan
        image = np.abs(func(image))
        image = np.uint8(np.rint(image.clip(0,1) * 255))
        return image

class Admin:
    def __init__(self,freq,tmp_mode,warp):
        self.warp = warp
        self.tmp_mode = tmp_mode
        self.system = None
        self.make_base_folder()
        self.freq = freq

    def set_system(self):
        '''set base directory -- \\ if Windows and / if Mac?'''
        path = os.getcwd()
        self.system = '\\'
        if '/' in path:
            self.system = '/'

    def make_base_folder(self):
        '''Make base folder for the images'''
        self.set_system()
        if not os.path.exists('bin_images'):
            os.mkdir('bin_images')
        elif os.path.exists('bin_images'):
            print("Warning, images may be overwritten!")

    def get_name(self,num_bits):
        name = self.make_folders(num_bits)
        im_bits = str(num_bits + 1)
        if im_bits == '8':
            im_bits = 'final_image'
        return (name + self.system + im_bits + ".png")

    def make_folders(self,num_bits):
        length = en(self.signals['a'])
        length //= 8
        new_folder = 'bin_images' + self.system + self.set_prefix() + str(length)
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)
        return new_folder

    def set_prefix(self):
        prefix = ''
        if self.warp[0] == 't':
            prefix += 'trig'
        if self.tmp_mode == 1: #1
            prefix += 'c'
        elif self.tmp_mode == 2: #2
            prefix += 's'
        elif self.tmp_mode == 3: #3
            prefix += 'u'
        elif self.tmp_mode == -1:
            prefix += 'sc'
       
        return prefix