from psychopy import visual

class artScreen:

    # this is a window that will draw an art on the screen

    def __init__(self,inputSize):
        self.size = inputSize
        self.window = visual.Window(size=self.size, pos=[960,540], fullscr=False,allowGUI=True, monitor="testMonitor")
        self.artImage = visual.ImageStim(self.window)
        self.artImage.draw(self.window)
        self.window.flip()
    
    def updateScreen(self,newImage):
        """update the art screen"""
        self.artImage.image = newImage
        self.artImage.draw(self.window)
        self.window.flip()
