from numpy import zeros, float64, sign

from .ball_stationary_event import BallStationaryEvent
from .ball_rest_event import BallRestEvent


class BallSpinningEvent(BallStationaryEvent):
    def __init__(self, t, i, r_0, omega_0_y, **kwargs):
        R = self.ball_radius
        self._omega_0_y = omega_0_y
        self._b = -5 * sign(omega_0_y) * self.mu_sp * self.g / (2 * R)
        T = abs(omega_0_y / self._b)
        super().__init__(t, i, r_0=r_0, T=T, **kwargs)
        self._next_motion_event = None
    
    @property
    def next_motion_event(self):
        if self._next_motion_event is None:
            self._next_motion_event = BallRestEvent(self.t + self.T, self.i, r_0=self._r_0)
        return self._next_motion_event
    
    def eval_angular_velocity(self, tau, out=None):
        if out is None:
            out = zeros(3, dtype=float64)
        else:
            out[:] = 0
        out[1] = self._omega_0_y + self._b * tau
        return out
    
    def json(self):
        d = super().json()
        d.update({
            'omega_0_y': self._omega_0_y,
            'b': self._b
        })
        return d