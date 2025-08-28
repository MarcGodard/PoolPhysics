
from numpy import zeros, array, empty, float64

from .ball_event import BallEvent


class BallMotionEvent(BallEvent):
    
    def __init__(self, t, i, T=None, a=None, b=None,
                 r_0=None, v_0=None, a_0=None,
                 omega_0=None,
                 **kwargs):
        """
        :param t: start time of event
        :param T: duration of event
        :param a: coefficients of the positional quadratic equation of motion (event-local time)
        :param b: coefficients of the angular velocity linear equation of motion (event-local time)
        :param r_0: ball position at start of event
        :param v_0: ball velocity at start of event
        :param omega_0: ball angular velocity at start of event
        """
        super().__init__(t, i, T=T, **kwargs)
        if a is None:
            a = zeros((3,3), dtype=float64)
        if b is None:
            b = zeros((2,3), dtype=float64)
        self._a = a
        self._b = b
        if r_0 is not None:
            a[0] = r_0
        if v_0 is not None:
            a[1] = v_0
        if a_0 is not None:
            a[2] = 0.5 * a_0
        if omega_0 is not None:
            b[0] = omega_0
        self._r_0 = a[0]
        self._v_0 = a[1]
        self._omega_0 = b[0]
        self._ab_global = None
        self._a_global = None
        self._next_motion_event = None
    
    @property
    def acceleration(self):
        return 2 * self._a[2]
    
    @property
    def next_motion_event(self):
        return self._next_motion_event
    
    @property
    def global_motion_coeffs(self):
        if self._ab_global is None:
            self._ab_global = self.calc_global_motion_coeffs(self.t, self._a, self._b)
        return self._ab_global[:3], self._ab_global[3:]
    
    @property
    def global_linear_motion_coeffs(self):
        if self._a_global is None:
            self._a_global = self.calc_global_linear_motion_coeffs(self.t, self._a)
        return self._a_global
    
    @staticmethod
    def calc_global_linear_motion_coeffs(t, a, out=None):
        """
        Calculates the coefficients of the global-time linear equations of motion.

        :param t: the global time of the start of the motion
        :param a: the local-time (0 at the start of the motion) linear motion coefficients
        """
        if out is None:
            out = a.copy()
        else:
            out[:] = a
        out[0] += -t * a[1] + t**2 * a[2]
        out[1] += -2 * t * a[2]
        return out
    
    def calc_shifted_motion_coeffs(self, t0):
        ab_global = self.calc_global_motion_coeffs(self.t - t0, self._a, self._b)
        return ab_global[:3], ab_global[3:]
    
    @staticmethod
    def calc_global_motion_coeffs(t, a, b, out=None):
        """
        Calculates the coefficients of the global-time equations of motion.

        :param t: the global time of the start of the motion
        :param a: the local-time (0 at the start of the motion) linear motion coefficients
        :param b: the local-time angular motion coefficients
        """
        if out is None:
            out = zeros((5,3), dtype=float64)
        out[:3] = a
        out[3:] = b
        a_global, b_global = out[:3], out[3:]
        a_global[0] += -t * a[1] + t**2 * a[2]
        a_global[1] += -2 * t * a[2]
        b_global[0] += -t * b[1]
        return out
    
    def eval_position(self, tau, out=None):
        if out is None:
            out = self._r_0.copy()
        else:
            out[:] = self._r_0
        if tau != 0:
            a = self._a
            out += tau * a[1] + tau**2 * a[2]
        return out
    
    def eval_velocity(self, tau, out=None):
        if out is None:
            out = self._v_0.copy()
        else:
            out[:] = self._v_0
        if tau != 0:
            out += 2 * tau * self._a[2]
        return out
    
    def eval_angular_velocity(self, tau, out=None):
        if out is None:
            out = empty(3, dtype=float64)
        out[:] = self._b[0] + tau * self._b[1]
        if self._b[0,1] >= 0:
            out[1] = max(0, out[1])
        else:
            out[1] = min(0, out[1])
        return out
    
    def eval_surface_velocity(self, tau, rd, v=None, omega=None, out=None):
        if v is None:
            v = self.eval_velocity(tau)
        if omega is None:
            omega = self.eval_angular_velocity(tau)
        if out is None:
            out = empty(3, dtype=float64)
        out[:] = v - self.ball_radius / sqrt(dot(rd, rd)) * cross(rd, omega)
        return out
    
    def eval_position_and_velocity(self, tau, out=None):
        if out is None:
            out = empty((2,3), dtype=float64)
        taus = array((1.0, tau, tau**2))
        a = self._a
        dot(taus, a, out=out[0])
        out[1] = a[1] + 2*tau*a[2]
        return out
    
    def __str__(self):
        return super().__str__()[:-1] + '\n r_0=%s\n v_0=%s\n a=%s\n omega_0=%s>' % (self._r_0, self._v_0, self.acceleration, self._omega_0)
    
    def json(self):
        d = super().json()
        d.update({
            'a': self._a.tolist(),
            'b': self._b.tolist()
        })
        return d
