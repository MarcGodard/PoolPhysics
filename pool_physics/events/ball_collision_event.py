from math import sqrt
from numpy import dot, array, float64

from .physics_event import PhysicsEvent
from .ball_rolling_event import BallRollingEvent
from .ball_sliding_event import BallSlidingEvent
from .ball_spinning_event import BallSpinningEvent
from .ball_rest_event import BallRestEvent


def printit(roots):
    return ',  '.join('%5.17f + %5.17fj' % (r.real, r.imag) if r.imag else '%5.17f' % r for r in roots)



class BallCollisionEvent(PhysicsEvent):
    
    def __init__(self, t, e_i, e_j):
        super().__init__(t)
        self.e_i, self.e_j = e_i, e_j
        self.i, self.j = e_i.i, e_j.i
        tau_i, tau_j = t - e_i.t, t - e_j.t
        self._r_i, self._r_j = e_i.eval_position(tau_i), e_j.eval_position(tau_j)
        self._v_i, self._v_j = e_i.eval_velocity(tau_i), e_j.eval_velocity(tau_j)
        self._r_ij = r_ij = self._r_j - self._r_i
        self._v_ij = v_ij = self._v_j - self._v_i
        self._y_loc = y_loc = 0.5 * r_ij / self.ball_radius
        self._x_loc = array((-y_loc[2], 0.0, y_loc[0]), dtype=float64)
        self._v_ij_y0 = dot(v_ij, y_loc)
        self._omega_i, self._omega_j = e_i.eval_angular_velocity(tau_i), e_j.eval_angular_velocity(tau_j)
    
    def __str__(self):
        return '<' + super().__str__()[:-1] + '''
 i,j = %s,%s
 r_i     = %s
 r_j     = %s
 v_i0    = %s   ||v_i0|| = %s
 v_j0    = %s   ||v_j0|| = %s
 v_ij_y0 = %s
 v_ij_y1 = %s
 v_i1    = %s   ||v_i1|| = %s
 v_j1    = %s   ||v_j1|| = %s
>>''' % (self.i, self.j,
         printit(self._r_i),
         printit(self._r_j),
         printit(self._v_i), sqrt(dot(self._v_i, self._v_i)),
         printit(self._v_j), sqrt(dot(self._v_j, self._v_j)),
         self._v_ij_y0,
         self._v_ij_y1,
         printit(self._v_i_1), sqrt(dot(self._v_i_1, self._v_i_1)),
         printit(self._v_j_1), sqrt(dot(self._v_j_1, self._v_j_1)))
    
    def __repr__(self):
        return '%s(%d,%d @ %s, v_ij_0=%s, v_ij_1=%s) at %x' % (self.__class__.__name__.split('.')[-1], self.i, self.j, self.t, self._v_ij_y0, self._v_ij_y1, id(self))
    
    def __lt__(self, other):
        if not isinstance(other, PhysicsEvent):
            return self.t < other
        return self.t <= other.t if not isinstance(other, BallCollisionEvent) \
            else self.t < other.t
    
    def __gt__(self, other):
        if not isinstance(other, PhysicsEvent):
            return self.t > other
        return self.t > other.t if not isinstance(other, BallCollisionEvent) \
            else self.t >= other.t
    
    @property
    def child_events(self):
        if self._child_events is None:
            child_events = []
            for (r, v_1, omega_1, e) in (
                    (self._r_i, self._v_i_1, self._omega_i_1, self.e_i),
                    (self._r_j, self._v_j_1, self._omega_j_1, self.e_j)
            ):
                if dot(v_1, v_1) == 0:
                    if abs(omega_1[1]) == 0:
                        e_1 = BallRestEvent(self.t, e.i,
                                            r_0=r,
                                            parent_event=self)
                    else:
                        e_1 = BallSpinningEvent(self.t, e.i,
                                                r_0=r,
                                                omega_0_y=omega_1[1],
                                                parent_event=self)
                else:
                    u_1 = v_1 + self.ball_radius * array((omega_1[2], 0.0, -omega_1[0]), dtype=float64)
                    if dot(u_1, u_1) == 0:
                        e_1 = BallRollingEvent(self.t, e.i,
                                               r_0=r,
                                               v_0=v_1,
                                               omega_0_y=omega_1[1],
                                               parent_event=self)
                    else:
                        e_1 = BallSlidingEvent(self.t, e.i,
                                               r_0=r,
                                               v_0=v_1,
                                               omega_0=omega_1,
                                               parent_event=self)
                child_events.append(e_1)
            self._child_events = tuple(child_events)
        return self._child_events
    
    def json(self):
        d = super().json()
        d.update({
            'i': self.i,
            'j': self.j
        })
        return d
