from psychopy import visual
from randomArt import randomArt

class artScreen:

    # this is a window that will draw an art on the screen

    def __init__(self,inputSize):
        self.size = inputSize
        self.window = visual.Window(size=self.size, pos=[self.size[0]//2,self.size[1]//2], fullscr=False,allowGUI=True, monitor="testMonitor")
        self.artImage = visual.ImageStim(self.window)
        self.visualize()

    def visualize(self):
        """visualize current screen"""
        self.artImage.draw(self.window)
        self.window.flip()
    
    def updateScreen(self,newImage):
        """update the art screen"""
        self.artImage.image = newImage
        self.visualize()

    def clearScreen(self):
        """clear the screen"""
        self.artImage = visual.ImageStim(self.window)
        self.visualize()

    def generateArt(self):
    # def generateArt(self,brainFeatures):
        """generates random art and updates the screen"""
        art = randomArt(self.size)
        updateScreen(art)
    #     art = randomArt(brainFeatures,self.size)
    #     updateScreen(art)

