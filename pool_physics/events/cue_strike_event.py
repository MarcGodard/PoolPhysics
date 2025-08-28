from math import sqrt
from numpy import dot, cross, array, float64

from .ball_event import BallEvent
from .ball_sliding_event import BallSlidingEvent


_k = array([0, 1, 0],        # upward-pointing basis vector :math:`\hat{k}`
           dtype=float64)    # of any ball-centered frame, following the convention of Marlow


class CueStrikeEvent(BallEvent):
    def __init__(self, t, i, r_i, r_c, V, M, q_i=None):
        """
        :param r_i: position of ball at moment of impact
        :param r_c: global coordinates of the point of contact
        :param V: cue velocity at moment of impact; the cue's velocity is assumed to be aligned with its axis
        :param M: cue mass
        :param q_i: rotation quaternion of ball at moment of impact
        """
        super().__init__(t, i)
        m, R, I = self.ball_mass, self.ball_radius, self.ball_I
        V = V.copy()
        V[1] = 0 # temporary: set vertical to 0
        self.V = V
        self.M = M
        self.Q = Q = r_c - r_i
        _j = -V; _j[1] = 0; _j /= sqrt(dot(_j, _j))
        _i = cross(_j, _k)
        a, b = dot(Q, _i), Q[1]
        c = sqrt(R**2 - a**2 - b**2)
        sin, cos = b/R, sqrt(R**2 - b**2)/R
        V_mag = sqrt(dot(V, V))
        F_mag = 2*m*V_mag / (
            1 + m/M + 5/(2*R**2)*(a**2 + b**2*cos**2 + c**2*sin**2 - 2*b*c*cos*sin)
        )
        omega_0 = ( (-c*F_mag*sin + b*F_mag*cos) * _i +
                                   (a*F_mag*sin) * _j +
                                  (-a*F_mag*cos) * _k ) / I
        self._child_events = (BallSlidingEvent(t, i,
                                               r_0=r_i,
                                               v_0=-F_mag/m*_j,
                                               omega_0=omega_0,
                                               q_0=q_i,
                                               parent_event=self),)
    
    @property
    def child_events(self):
        return self._child_events
    
    def __str__(self):
        return super().__str__()[:-1] + '\n Q=%s\n V=%s\n M=%s>' % (self.Q, self.V, self.M)
    
    def json(self):
        d = super().json()
        d.update({
            'V': self.V.tolist(),
            'M': self.M,
            'Q': self.Q.tolist()
        })
        return d
