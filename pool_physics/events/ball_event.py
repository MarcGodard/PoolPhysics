from .physics_event import PhysicsEvent


class BallEvent(PhysicsEvent):
    def __init__(self, t, i, **kwargs):
        super().__init__(t, **kwargs)
        self.i = i
    
    @property
    def next_motion_event(self):
        return None
    
    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.t == other.t and self.T == other.T and self.i == other.i
    
    def __str__(self):
        return super().__str__()[:-1] + " i=%d>" % self.i
    
    def __repr__(self):
        return '%s(%d @ %s) at %x' % (self.__class__.__name__.split('.')[-1], self.i, self.t, id(self))
    
    def json(self):
        d = super().json()
        d.update({
            'i': self.i
        })
        return d