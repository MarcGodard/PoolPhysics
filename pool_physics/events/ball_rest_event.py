from numpy import zeros, float64

from .ball_stationary_event import BallStationaryEvent


class BallRestEvent(BallStationaryEvent):
    
    def __init__(self, t, i, **kwargs):
        super().__init__(t, i, T=float('inf'), **kwargs)
    
    def eval_angular_velocity(self, tau, out=None):
        if out is None:
            out = zeros(3, dtype=float64)
        else:
            out[:] = 0
        return out