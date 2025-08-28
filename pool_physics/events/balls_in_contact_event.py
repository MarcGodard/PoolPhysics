from .physics_event import PhysicsEvent


class BallsInContactEvent(PhysicsEvent):

    def __init__(self, e_i, e_j):
        assert e_i.t == e_j.t
        t = e_i.t
        T = min(e_i.T, e_j.T)
        super().__init__(t, T=T)
        self.i, self.j = e_i.i, e_j.i
        e_i._parent_event = self
        e_j._parent_event = self
        nme = e_i.next_motion_event
        while nme:
            nme._parent_event = self
            nme = nme.next_motion_event
        nme = e_j.next_motion_event
        while nme:
            nme._parent_event = self
            nme = nme.next_motion_event
        self._child_events = (e_i, e_j)

    @property
    def child_events(self):
        return self._child_events

    def __str__(self):
        return super().__str__()[:-1] + '''
 i,j = %s,%s
 e_i = %s
 e_j = %s
>''' % (self.i, self.j, *self._child_events)
