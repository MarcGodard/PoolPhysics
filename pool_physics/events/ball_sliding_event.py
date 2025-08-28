
from math import sqrt
from numpy import dot, array, float64, sign

from .ball_motion_event import BallMotionEvent
from .ball_rolling_event import BallRollingEvent

class BallSlidingEvent(BallMotionEvent):
    def __init__(self, t, i, r_0, v_0, omega_0, **kwargs):
        R, mu_s, g = self.ball_radius, self.mu_s, self.g
        u_0 = v_0 + R * array((omega_0[2], 0.0, -omega_0[0]), dtype=float64)
        u_0_mag = sqrt(dot(u_0, u_0))
        T = 2 * u_0_mag / (7 * mu_s * g)
        super().__init__(t, i, T=T, r_0=r_0, v_0=v_0, omega_0=omega_0, **kwargs)
        self._u_0 = u_0
        self._u_0_mag = u_0_mag
        self._a[2] = -0.5 * mu_s * g * u_0 / u_0_mag
        self._b[0] = omega_0
        self._b[1,::2] = 5 * mu_s * g / (2 * R) / u_0_mag * array((u_0[2], -u_0[0]), dtype=float64)
        self._b[1,1] = -sign(omega_0[1]) * 5 * self.mu_sp * g / (2 * R)
        self._next_motion_event = None
    @property
    def next_motion_event(self):
        if self._next_motion_event is None:
            i, t, T = self.i, self.t, self.T
            omega_1 = self.eval_angular_velocity(T)
            self._next_motion_event = BallRollingEvent(t + T, i,
                                                       r_0=self.eval_position(T),
                                                       v_0=self.eval_velocity(T),
                                                       omega_0_y=omega_1[1])
        return self._next_motion_event
