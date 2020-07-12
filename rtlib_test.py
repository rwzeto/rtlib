import sys
sys.path.append('./target/debug/')
import rtlib

class ray():
    def __init__(self, rray=None):
        if rray is None:
            self.power = 0.0
            self.pos = [0.0, 0.0, 0.0] 
            self.dir = [0.0, 0.0, 0.0] 
            self.color = [0.0, 0.0, 0.0] 
            self.shadow = False
            self._rray = None
        else:
            self.power = rray.ppower
            self.pos = rray.ppos
            self.dir = rray.pdir
            self.color = rray.pcolor
            self.shadow = rray.pshadow
            self._rray = rray

    def _initialize_rtlib_ray(self):
        self._rray = rtlib.rray()
        self._rray.ppos = self.pos
        self._rray.pdir = self.dir
        self._rray.pcolor = self.color
        self._rray.pshadow = self.shadow
    def propagate(self):
        self._initialize_rtlib_ray()
        return ray(rray=self._rray.propagate())
    

rayone = ray()
rayone.dir = [1.0,0.0,0.0]
raytwo = rayone.propagate()

print(raytwo.color)