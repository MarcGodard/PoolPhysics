from numpy import dot

from .ball_collision_event import BallCollisionEvent
from ..collisions import collide_balls, collide_balls_f90

class SimulatedBallCollisionEvent(BallCollisionEvent):
    
    collide_balls = staticmethod(collide_balls)
    
    def __init__(self, t, e_i, e_j):
        super().__init__(t, e_i, e_j)
        r_i, r_j = self._r_i, self._r_j
        v_i, v_j = self._v_i, self._v_j
        omega_i, omega_j = self._omega_i, self._omega_j
        y_loc = self._y_loc
        self._v_i_1, self._omega_i_1, self._v_j_1, self._omega_j_1 = \
            self.collide_balls(
                r_i, v_i, omega_i, r_j, v_j, omega_j,
                self.ball_mass*abs(self._v_ij_y0)/6400
            )
        self._v_ij_y1 = dot(self._v_j_1 - self._v_i_1, y_loc)
        self._child_events = None


class FSimulatedBallCollisionEvent(SimulatedBallCollisionEvent):
    collide_balls = staticmethod(collide_balls_f90)
