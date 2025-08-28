from math import sqrt
from numpy import dot, cross, zeros, array, empty, float64, sign

from .ball_collision_event import BallCollisionEvent
from .ball_rest_event import BallRestEvent
from .ball_spinning_event import BallSpinningEvent
from .ball_rolling_event import BallRollingEvent
from .ball_sliding_event import BallSlidingEvent

_k = array([0, 1, 0],        # upward-pointing basis vector :math:`\hat{k}`
           dtype=float64)    # of any ball-centered frame, following the convention of Marlow


class SimpleBallCollisionEvent(BallCollisionEvent):

    def __init__(self, t, e_i, e_j, v_factor=0.98):
        """Simple one-parameter elastic collision model with no friction between balls or any other surface."""
        super().__init__(t, e_i, e_j)
        y_loc = self._y_loc
        v_i, v_j = self._v_i, self._v_j
        v_ix = dot(v_i, y_loc) * y_loc
        v_jx = dot(v_j, y_loc) * y_loc
        v_iy = v_i - v_ix
        v_jy = v_j - v_jx
        v_ix_1 = 0.5 * ((1 - v_factor) * v_ix + (1 + v_factor) * v_jx)
        v_jx_1 = 0.5 * ((1 - v_factor) * v_jx + (1 + v_factor) * v_ix)
        v_i_1 = v_iy + v_ix_1
        v_j_1 = v_jy + v_jx_1
        self._v_i_1, self._v_j_1 = v_i_1, v_j_1
        self._v_ij_y1 = dot(v_j_1 - v_i_1, y_loc)
        omega_i, omega_j = self._omega_i, self._omega_j
        self._omega_i_1, self._omega_j_1 = omega_i.copy(), omega_j.copy()
        self._child_events = None
    
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
                elif isinstance(e, BallSlidingEvent) \
                     and abs(dot(v_1, self._y_loc) / self.ball_radius) > abs(dot(omega_1, cross(_k, self._y_loc))):
                    e_1 = BallSlidingEvent(self.t, e.i,
                                           r_0=r,
                                           v_0=v_1,
                                           omega_0=omega_1,
                                           parent_event=self)
                else:
                    e_1 = BallRollingEvent(self.t, e.i,
                                           r_0=r,
                                           v_0=v_1,
                                           omega_0_y=omega_1[1],
                                           parent_event=self)
                child_events.append(e_1)
            self._child_events = tuple(child_events)
        return self._child_events

