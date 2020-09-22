import numpy as np
from PIL import Image
import random
import pdb
import cv2
import matplotlib.pyplot as plt


def circleArt(imageArray,freqNoise,freqState,artFeatures):

    ''' make an image in the HSV system based on the given data
    
    size is a tuple - the size of the image
    freqNoise is a dict, storing 'True' if there is noise and 'False' if there is not noise, for each bandwidth of brain wave for the last second of data
    freqState is a dict, storing the amplitude level (low or high) of each bandwidth of brain wave for the last second of data
    artFeatures is a dict, storing the art features (hue, saturation, value, line quality) assigned to each bandwidth of brain wave

    '''

    # the bandwidths in the order of presentation in the Main Window
    freqOrder = ['Beta','Alpha','Theta','Delta']

    # unpack size into x and y dimensions
    xdim,ydim = imageArray.shape[:2]

    lineQualBandwidth = freqOrder[artFeatures.index(3)]

    # pdb.set_trace()
    if not freqNoise[lineQualBandwidth]:
        if freqState[lineQualBandwidth] == 'High':
            thicknessRange = (int(xdim*0.3),int(xdim*0.4))
            nCircles = random.randint(20,30)
            alpha = random.uniform(0.6,1)
            
        else:
            thicknessRange = (int(xdim*0.1),int(xdim*0.2))
            nCircles = random.randint(5,10)
            alpha = random.uniform(0.2, 0.4)
    else:
        thicknessRange = (5,int(xdim*0.05))
        nCircles = random.randint(0,3)
        alpha = random.uniform(0,0.1)

    rBandwidth = freqOrder[artFeatures.index(0)]
    if not freqNoise[rBandwidth]:
        if freqState[rBandwidth] == 'High':
            rRange = (200,255)
        else:
            rRange = (100, 180)
    else:
        rRange = (30,60)

    gBandwidth = freqOrder[artFeatures.index(1)]
    if not freqNoise[gBandwidth]:
        if freqState[gBandwidth] == 'High':
            gRange = (200,255)
        else:
            gRange = (100, 180)
    else:
        gRange = (30,60)

    bBandwidth = freqOrder[artFeatures.index(2)]
    if not freqNoise[bBandwidth]:
        if freqState[bBandwidth] == 'High':
            bRange = (200,255)
        else:
            bRange = (100, 180)
    else:
        bRange = (30,60)

    for i in range(nCircles):
        random.choice([
            cv2.circle(imageArray,(random.randint(int(xdim*0.2),int(xdim*0.8)),random.randint(int(ydim*0.2),int(ydim*0.8))),random.randint(int(xdim*0.1),int(xdim*0.5)),(random.randint(rRange[0],rRange[1]),random.randint(gRange[0],gRange[1]),random.randint(bRange[0],bRange[1]))),
            cv2.circle(imageArray,(random.randint(int(xdim*0.2),int(xdim*0.8)),random.randint(int(ydim*0.2),int(ydim*0.8))),random.randint(int(xdim*0.1),int(xdim*0.5)),(random.randint(rRange[0],rRange[1]),random.randint(gRange[0],gRange[1]),random.randint(bRange[0],bRange[1])),thickness = random.randint(thicknessRange[0],thicknessRange[1]))
            ])

        imageArray = cv2.addWeighted(imageArray, alpha, imageArray, 1 - alpha, 0)

    # return image array
    return imageArray
    

def main():

    size=[800,800]

    # initialize an array for the image -- x by y arrays of 3-item arrays for the HSV values of each pixel in the image
    imageArray = np.zeros((size[0],size[1],3),dtype=np.uint8)
    # initialize an x by y array for making patterns on the image by using blend modes
    # imageBlend = np.zeros((size[0], size[1]), dtype=np.uint8)

    freqNoise = {
        'Delta': False,
        'Theta': False,
        'Alpha': False,
        'Beta': False
    }

    freqState = {
        'Delta': 'Low',
        'Theta': 'Low',
        'Alpha': 'Low',
        'Beta': 'High'
    }

    # image.show()
    print('freqNoise:\n',freqNoise)
    print('freqState:\n',freqState)

    artFeatures = [0,1,2,3]
    
    image = circleArt(imageArray,freqNoise,freqState,artFeatures)

    from PIL import Image
    import math

    preimg = Image.fromarray(image,mode='RGB')
    preimg.show()

    moodArray = np.array([0.7,0.2,0.1])

    # the image that will be blended into the existing image array
    # blendLayer = np.zeros((size[0],size[1],4), dtype=np.uint8)

    if moodArray.max() > 0.6:   # only blend if we have a clear enough classification, i.e. probability above 0.6

        mood = np.where(moodArray == moodArray.max())[0]   # find which mood had the largest probability

        # if (mood[0] == 2)[0]:   # change the blend layer to all white only if the mood is pos
        #     blendLayer.fill(255)

        # blendLayer[:,:,3] = random.randint(100,200)   # randomly generate an alpha value for the blend layer
        
        # newImage = np.zeros((size[0],size[1],4), dtype=np.uint8)
        # newImage[:,:,0:3]= image
        # newImage[:,:,3] = 255

        # # alpha blending (https://en.wikipedia.org/wiki/Alpha_compositing#Alpha_blending)
        # image = Image.alpha_composite(Image.fromarray(
        #     newImage, mode='RGBA'), Image.fromarray(blendLayer, mode='RGBA'))

        for i in range(size[0]):
            for j in range(size[1]):
                offset_x = int(25.0 * math.sin(2 * 3.14 * i / 180))
                if i+offset_x < size[0]:
                    image[i,j] = image[(i+offset_x),j]
                else:
                    image[i,j] = 0

        image = Image.fromarray(image,mode='RGB')
        image.show()

if __name__ == '__main__':
    main()
