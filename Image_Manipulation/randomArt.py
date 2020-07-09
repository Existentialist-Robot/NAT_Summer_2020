import numpy as np
import random
from PIL import Image
import pdb

def randomArt(size):
# with the brain features, will take in the features as arguments:
# def randomArt(size,feature1,feature2,...)

    '''This function generates a random art image

    size is a 2x1 list with the x and y dimensions of the image'''
    
    dX, dY = size
    xArray = np.linspace(0.0, 1.0, dX).reshape((1, dX, 1))
    yArray = np.linspace(0.0, 1.0, dY).reshape((dY, 1, 1))

    # TODO: add translation of brain features to some sort of input into the image creating process

    def randColor():
        return np.array([random.random(), random.random(), random.random()]).reshape((1, 1, 3))
    def getX(): return xArray
    def getY(): return yArray
    def safeDivide(a, b):
        return np.divide(a, np.maximum(b, 0.001))

    functions = [(0, randColor),
                (0, getX),
                (0, getY),
                (1, np.sin),
                (1, np.cos),
                (2, np.add),
                (2, np.subtract),
                (2, np.multiply),
                (2, safeDivide)]
    depthMin = 2
    depthMax = 10

    def buildImg(depth = 0):
        funcs = [f for f in functions if
                    (f[0] > 0 and depth < depthMax) or
                    (f[0] == 0 and depth >= depthMin)]
        nArgs, func = random.choice(funcs)
        args = [buildImg(depth + 1) for n in range(nArgs)]
        return func(*args)

    img = buildImg()
    # pdb.set_trace()

    # Ensure it has the right dimensions, dX by dY by 3
    img = np.tile(img, (dX / img.shape[0], dY / img.shape[1], 3 / img.shape[2]))

    # Convert to 8-bit, send to PIL and return
    img8Bit = np.uint8(np.rint(img.clip(0.0, 1.0) * 255.0))
    return Image.fromarray(img8Bit)