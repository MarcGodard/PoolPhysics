from numpy import zeros, float64

from .ball_event import BallEvent

class BallStationaryEvent(BallEvent):
    def __init__(self, t, i, r_0=None, **kwargs):
        super().__init__(t, i, **kwargs)
        if r_0 is None:
            r_0 = zeros(3, dtype=float64)
        self._r_0 = self._r = r_0
        self._a_global = None
    
    @property
    def acceleration(self):
        return zeros(3, dtype=float64)
    
    @property
    def global_motion_coeffs(self):
        if self._a_global is None:
            self._a_global = a = zeros((3,3), dtype=float64)
            a[0] = self._r_0
        return self._a_global, None
    
    @property
    def global_linear_motion_coeffs(self):
        if self._a_global is None:
            self._a_global = a = zeros((3,3), dtype=float64)
            a[0] = self._r_0
        return self._a_global
    
    def calc_shifted_motion_coeffs(self, t0):
        return self.global_motion_coeffs
    
    def eval_position(self, tau, out=None):
        if out is None:
            out = self._r_0.copy()
        else:
            out[:] = self._r_0
        return out
    
    def eval_velocity(self, tau, out=None):
        if out is None:
            out = zeros(3, dtype=float64)
        else:
            out[:] = 0
        return out
    
    def eval_slip_velocity(self, tau, out=None):
        if out is None:
            out = zeros(3, dtype=float64)
        else:
            out[:] = 0
        return out
    
    def __str__(self):
        return super().__str__()[:-1] + '\n r=%s>' % self._r
    
    def json(self):
        """Returns a dict which can be JSON-serialized (by json.dumps)"""
        d = super().json()
        d.update({
            'r_0': self._r_0.tolist()
        })
        return d