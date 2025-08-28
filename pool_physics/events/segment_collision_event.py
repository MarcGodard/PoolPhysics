from numpy import dot

from .ball_event import BallEvent
from .ball_sliding_event import BallSlidingEvent
from .ball_rolling_event import BallRollingEvent


class SegmentCollisionEvent(BallEvent):
    kappa = 0.6
    
    def __init__(self, t, e_i, seg, nor, tan):
        super().__init__(t, e_i.i)
        self.e_i = e_i
        self.seg = seg
        self.nor = nor
        self.tan = tan
        tau = t - e_i.t
        r = e_i.eval_position(tau)
        self.r = r
        self.r_c = r - self.ball_radius * nor
        self._child_events = None
    
    @property
    def child_events(self):
        if self._child_events is None:
            R = self.ball_radius
            e_i = self.e_i
            tau = self.t - e_i.t
            v_1 = e_i.eval_velocity(tau)
            omega_1 = e_i.eval_angular_velocity(tau)
            v_1 = -self.kappa * dot(v_1, self.nor) * self.nor + dot(v_1, self.tan) * self.tan
            omega_1[:] = 0
            if isinstance(e_i, BallSlidingEvent) \
               and abs(dot(v_1, self.tan)) / R <= abs(dot(omega_1, self.nor)):
                self._child_events = (BallSlidingEvent(self.t, e_i.i,
                                                       r_0=self.r,
                                                       v_0=v_1,
                                                       omega_0=omega_1,
                                                       parent_event=self),)
            else:
                self._child_events = (BallRollingEvent(self.t, e_i.i,
                                                       r_0=self.r,
                                                       v_0=v_1,
                                                       parent_event=self),)
        return self._child_events
    
    def __str__(self):
        return super().__str__()[:-1] + " seg=%d r=%s r_c=%s>" % (self.seg, self.r, self.r_c)
    
    def json(self):
        d = super().json()
        d.update({
            'e_i': self.e_i.json(),
            'seg': self.seg,
            'nor': self.nor.tolist(),
            'tan': self.tan.tolist()
        })
        return d

