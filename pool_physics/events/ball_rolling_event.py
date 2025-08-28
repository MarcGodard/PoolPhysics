
from math import sqrt
from numpy import dot, zeros, array, float64, sign

from .ball_motion_event import BallMotionEvent
from .ball_rest_event import BallRestEvent
from .ball_spinning_event import BallSpinningEvent


class BallRollingEvent(BallMotionEvent):
    def __init__(self, t, i, r_0, v_0, omega_0_y=0.0, **kwargs):
        R = self.ball_radius
        v_0_mag = sqrt(dot(v_0, v_0))
        T = v_0_mag / (self.mu_r * self.g)
        omega_0 = array((v_0[2]/R, omega_0_y, -v_0[0]/R), dtype=float64)
        super().__init__(t, i, T=T, r_0=r_0, v_0=v_0, omega_0=omega_0, **kwargs)
        self._a[2] = -0.5 * self.mu_r * self.g * v_0 / v_0_mag
        self._b[1,::2] = -omega_0[::2] / T
        self._b[1,1] = -sign(omega_0_y) * 5 / 7 * self.mu_r * self.g / R
        #self._b[1,1] = -sign(omega_0_y) * 5 / 2 * self.mu_sp * self.g / R
        self._next_motion_event = None
    @property
    def next_motion_event(self):
        if self._next_motion_event is None:
            i, t, T = self.i, self.t, self.T
            omega_1 = self.eval_angular_velocity(T)
            if abs(omega_1[1]) < self._ZERO_TOLERANCE:
                self._next_motion_event = BallRestEvent(t + T, i, r_0=self.eval_position(T))
            else:
                self._next_motion_event = BallSpinningEvent(t + T, i, r_0=self.eval_position(T),
                                                            omega_0_y=omega_1[1])
        return self._next_motion_event
    def eval_slip_velocity(self, tau, out=None, **kwargs):
        if out is None:
            out = zeros(3, dtype=float64)
        else:
            out[:] = 0
        return out
