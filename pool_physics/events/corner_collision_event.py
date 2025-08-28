from math import sqrt
from numpy import dot, array, float64

from .ball_event import BallEvent
from .ball_sliding_event import BallSlidingEvent
from .ball_rolling_event import BallRollingEvent

_k = array([0, 1, 0],        # upward-pointing basis vector :math:`\hat{k}`
           dtype=float64)    # of any ball-centered frame, following the convention of Marlow


class CornerCollisionEvent(BallEvent):
    kappa = 0.6 # coefficient of restitution

    def __init__(self, t, e_i, i_c, r_c):
        super().__init__(t, e_i.i)
        self.e_i = e_i
        self.i_c = i_c
        self.r_c = r_c.copy()
        tau = self.t - e_i.t
        self.r_i = r_i = e_i.eval_position(tau)
        v_0 = self.v_0 = e_i.eval_velocity(tau)
        j_loc = self.r_c - r_i
        j_loc[1] = 0
        j_loc /= -sqrt(dot(j_loc, j_loc))
        self.j_loc = j_loc
        i_loc = array((-j_loc[2], 0.0, j_loc[0]), dtype=float64)
        self.i_loc = i_loc
        v_1 = v_0
        omega_1 = self.omega_0 = e_i.eval_angular_velocity(tau)
        v_1x,  v_1y  =  dot(v_1, i_loc),  -self.kappa * dot(v_1, j_loc)
        self.v_1 = v_1 = v_1x * i_loc \
                       + v_1y * j_loc
        self.omega_1 = dot(omega_1, j_loc) * j_loc \
                     + omega_1[1] * _k \
                     - v_1y / self.ball_radius * i_loc
        self._child_events = None
    
    @property
    def child_events(self):
        if self._child_events is None:
            R = self.ball_radius
            u_1 = self.v_1 + R * array((self.omega_1[2], 0.0, -self.omega_1[0]), dtype=float64)
            if isinstance(self.e_i, BallSlidingEvent) \
               and dot(u_1, u_1) > 0:
                self._child_events = (BallSlidingEvent(self.t, self.e_i.i,
                                                       r_0=self.r_i,
                                                       v_0=self.v_1,
                                                       omega_0=self.omega_1,
                                                       parent_event=self),)
            else:
                self._child_events = (BallRollingEvent(self.t, self.e_i.i,
                                                       r_0=self.r_i,
                                                       v_0=self.v_1,
                                                       omega_0_y=self.omega_1[1],
                                                       parent_event=self),)
        return self._child_events
    
    def __str__(self):
        return super().__str__()[:-1] + " i_c=%s r_c=%s v_0=%s v_1=%s omega_0=%s omega_1=%s>" % (
            self.i_c, self.r_c, self.v_0, self.v_1, self.omega_0, self.omega_1)
    
    def json(self):
        d = super().json()
        d.update({
            'i_c': self.i_c,
            'r_c': self.r_c.tolist()
        })
        return d
