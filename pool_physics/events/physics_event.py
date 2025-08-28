from numpy import sin, cos, array, empty, float64

class PhysicsEvent(object):
    ball_radius = 0.02625
    ball_mass = 0.1406
    ball_I = 2/5 * ball_mass * ball_radius**2
    mu_s = 0.21
    mu_b = 0.05
    e = 0.89
    mu_r = 0.016 # coefficient of rolling friction between ball and table
    mu_sp = 0.044 # coefficient of spinning friction between ball and table
    g = 9.81 # magnitude of acceleration due to gravity
    _ZERO_TOLERANCE = 1e-8
    _ZERO_TOLERANCE_SQRD = _ZERO_TOLERANCE**2
    
    def __init__(self, t, T=0.0, parent_event=None, **kwargs):
        """
        Base class of pool physics events.

        :param t: time of event start
        :param T: time duration of the event (default is 0, i.e. instantaneous)
        """
        self.t = t
        self.T = T
        self._parent_event = parent_event
        self.interval = array((t, t + T))
    
    @property
    def child_events(self):
        return ()
    
    @property
    def parent_event(self):
        return self._parent_event
    
    @staticmethod
    def set_quaternion_from_euler_angles(psi=0.0, theta=0.0, phi=0.0, out=None):
        if out is None: out = empty(4, dtype=float64)
        angles = array((psi, theta, phi))
        c1, c2, c3 = cos(0.5 * angles)
        s1, s2, s3 = sin(0.5 * angles)
        out[0] = s1*c2*c3 + c1*s2*s3
        out[1] = c1*s2*c3 - s1*c2*s3
        out[2] = c1*c2*s3 + s1*s2*c3
        out[3] = c1*c2*c3 - s1*s2*s3
        return out
    
    @staticmethod
    def events_str(events, sep='\n\n' + 48*'-' + '\n\n'):
        return sep.join('%3d: %s' % (i_e, e) for i_e, e in enumerate(events))
    
    def __lt__(self, other):
        if isinstance(other, PhysicsEvent):
            return self.t < other.t
        else:
            return self.t < other
    
    def __gt__(self, other):
        if isinstance(other, PhysicsEvent):
            return self.t > other.t
        else:
            return self.t > other
    
    def __str__(self):
        if self.T == 0.0 or self.T == float('inf'):
            return '<%16s ( %5.15f )>' % (self.__class__.__name__, self.t)
        else:
            return '<%16s ( %5.15f ,  %5.15f )>' % (self.__class__.__name__, self.t, self.t+self.T)
    
    def json(self):
        return {
            'type': self.__class__.__name__,
            't': self.t,
            'T': self.T,
        }
