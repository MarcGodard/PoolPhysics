
from numpy import dot, array, float64, sign

from .ball_event import BallEvent
from .ball_sliding_event import BallSlidingEvent
from .ball_rolling_event import BallRollingEvent

class RailCollisionEvent(BallEvent):
    _J_LOC = array((
        ( 0.0,  0.0, -1.0),
        ( 1.0,  0.0,  0.0),
        ( 0.0,  0.0,  1.0),
        (-1.0,  0.0,  0.0)), dtype=float64)
    _J_VAR = [2, 0, 2, 0]
    _I_LOC = array((
        ( 1.0,  0.0,  0.0),
        ( 0.0,  0.0,  1.0),
        (-1.0,  0.0,  0.0),
        ( 0.0,  0.0, -1.0)), dtype=float64)
    kappa = 0.6 # coefficient of restitution
    
    def __init__(self, t, e_i, side):
        super().__init__(t, e_i.i)
        self.e_i = e_i
        self.side = side
        self._child_events = None
    
    @property
    def child_events(self):
        if self._child_events is None:
            R = self.ball_radius
            e_i = self.e_i
            tau = self.t - e_i.t
            v_1 = e_i.eval_velocity(tau)
            omega_1 = e_i.eval_angular_velocity(tau)
            side = self.side
            j_var = self._J_VAR[side]
            i_var = 2 - j_var
            v_1[j_var] *= -self.kappa
            omega_1[j_var] *= 0.8
            omega_1[i_var] = -sign(omega_1[i_var]) * abs(v_1[j_var]) / R
            if isinstance(e_i, BallSlidingEvent) \
               and abs(dot(v_1, self._J_LOC[side])) / R <= abs(dot(omega_1, self._I_LOC[side])):
                self._child_events = (BallSlidingEvent(self.t, e_i.i,
                                                       r_0=e_i.eval_position(tau),
                                                       v_0=v_1,
                                                       omega_0=omega_1,
                                                       parent_event=self),)
            else:
                self._child_events = (BallRollingEvent(self.t, e_i.i,
                                                       r_0=e_i.eval_position(tau),
                                                       v_0=v_1,
                                                       parent_event=self),)
        return self._child_events
    
    def __str__(self):
        return super().__str__()[:-1] + " side=%d>" % self.side
    
    def json(self):
        d = super().json()
        d.update({
            'e_i': self.e_i.json(),
            'side': self.side
        })
        return d
