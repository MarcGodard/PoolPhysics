"""
Collision correction system for pool physics engine.

This module implements a penetration resolution system that detects ball
overlaps and corrects them by stepping back in time and adjusting positions
to prevent penetrations from occurring.
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Dict
import copy
from .events import BallMotionEvent, BallCollisionEvent

_logger = logging.getLogger(__name__)


class PenetrationCorrector:
    """
    Corrects ball penetrations by stepping back in time and adjusting positions.
    
    When a penetration is detected, this system:
    1. Steps back to just before the penetration occurred
    2. Adjusts ball positions to maintain separation
    3. Continues simulation from the corrected state
    """
    
    def __init__(self, ball_radius: float, min_separation: float = None):
        self.ball_radius = ball_radius
        self.min_separation = min_separation or (2 * ball_radius + 1e-6)  # Minimum distance between ball centers
        self.correction_tolerance = 1e-9
        self.max_correction_attempts = 10
        
    def detect_penetration(self, ball_positions: List[np.ndarray]) -> Optional[Tuple[int, int, float]]:
        """
        Detect if any balls are penetrating.
        
        Returns (ball_i, ball_j, penetration_depth) or None if no penetration.
        """
        for i in range(len(ball_positions)):
            for j in range(i + 1, len(ball_positions)):
                distance = np.linalg.norm(ball_positions[j] - ball_positions[i])
                if distance < self.min_separation:
                    penetration_depth = self.min_separation - distance
                    return i, j, penetration_depth
        
        return None
    
    def correct_penetration(self, physics_engine, penetration_time: float, 
                          ball_i: int, ball_j: int) -> bool:
        """
        Correct a detected penetration by stepping back and adjusting positions.
        
        Returns True if correction was successful, False otherwise.
        """
        _logger.info(f'Correcting penetration between balls {ball_i} and {ball_j} at time {penetration_time}')
        
        # Simple approach: Step back a small amount and separate balls
        step_back_time = max(0, penetration_time - 0.001)  # Step back 1ms
        
        # Get ball positions at step-back time
        pos_i = self._get_ball_position_at_time(physics_engine, ball_i, step_back_time)
        pos_j = self._get_ball_position_at_time(physics_engine, ball_j, step_back_time)
        
        if pos_i is None or pos_j is None:
            _logger.warning('Could not get ball positions for correction')
            return False
        
        # Check if balls are still penetrating at step-back time
        distance = np.linalg.norm(pos_j - pos_i)
        if distance < self.min_separation:
            # Still penetrating, need to separate them
            separation_vector = pos_j - pos_i
            if np.linalg.norm(separation_vector) < 1e-12:
                # Balls at same position, use default separation
                separation_vector = np.array([self.min_separation, 0, 0])
            else:
                # Normalize and scale to minimum separation
                separation_vector = separation_vector / np.linalg.norm(separation_vector) * self.min_separation
            
            # Adjust positions
            center = (pos_i + pos_j) / 2
            new_pos_i = center - separation_vector / 2
            new_pos_j = center + separation_vector / 2
            
            # Apply corrections by creating new motion events
            success = self._create_corrected_motion_events(physics_engine, ball_i, ball_j, 
                                                         new_pos_i, new_pos_j, step_back_time)
            if success:
                # Rewind physics to step-back time
                physics_engine.t = step_back_time
                _logger.info(f'Successfully corrected penetration, rewound to time {step_back_time}')
                return True
        
        _logger.warning('Penetration correction failed')
        return False
    
    def _find_safe_time(self, physics_engine, ball_i: int, ball_j: int, 
                       penetration_time: float) -> Optional[float]:
        """
        Find the last time when balls were safely separated.
        """
        # Binary search for safe time
        t_start = max(0, penetration_time - 0.1)  # Look back up to 100ms
        t_end = penetration_time
        
        for _ in range(20):  # Max 20 iterations
            t_mid = (t_start + t_end) / 2
            
            # Get ball positions at t_mid
            pos_i = self._get_ball_position_at_time(physics_engine, ball_i, t_mid)
            pos_j = self._get_ball_position_at_time(physics_engine, ball_j, t_mid)
            
            if pos_i is None or pos_j is None:
                t_start = t_mid
                continue
            
            distance = np.linalg.norm(pos_j - pos_i)
            
            if distance >= self.min_separation:
                t_start = t_mid  # Safe time, can go later
            else:
                t_end = t_mid    # Penetration, must go earlier
            
            if (t_end - t_start) < self.correction_tolerance:
                break
        
        # Return the safe time with a small buffer
        return t_start - 1e-6 if t_start > 1e-6 else 0
    
    def _get_ball_position_at_time(self, physics_engine, ball_id: int, time: float) -> Optional[np.ndarray]:
        """Get ball position at specific time by evaluating motion events."""
        # Find the active motion event for this ball at the given time
        for event in physics_engine.ball_events[ball_id]:
            if event.t <= time <= event.t + event.T:
                tau = time - event.t
                return event.eval_position(tau)
        
        return None
    
    def _rewind_physics(self, physics_engine, target_time: float):
        """
        Rewind physics simulation to target time.
        """
        # Store current state
        original_time = physics_engine.t
        
        # Reset physics to target time
        physics_engine.t = target_time
        
        # Remove all events that occur after target time
        physics_engine.events = [e for e in physics_engine.events if e.t <= target_time]
        
        # Update ball motion events
        for ball_id in physics_engine._ball_motion_events:
            event = physics_engine._ball_motion_events[ball_id]
            if event.t > target_time:
                # Find previous event for this ball
                previous_events = [e for e in physics_engine.ball_events[ball_id] if e.t <= target_time]
                if previous_events:
                    physics_engine._ball_motion_events[ball_id] = previous_events[-1]
                else:
                    # Remove ball from active motion if no valid event
                    del physics_engine._ball_motion_events[ball_id]
            elif event.t + event.T > target_time:
                # Truncate event duration to target time
                event.T = target_time - event.t
        
        _logger.debug(f'Rewound physics from {original_time:.6f} to {target_time:.6f}')
    
    def _apply_position_correction(self, physics_engine, ball_i: int, ball_j: int, 
                                 correction_time: float) -> bool:
        """
        Apply position correction to prevent penetration.
        """
        # Get current positions
        pos_i = self._get_ball_position_at_time(physics_engine, ball_i, correction_time)
        pos_j = self._get_ball_position_at_time(physics_engine, ball_j, correction_time)
        
        if pos_i is None or pos_j is None:
            return False
        
        # Calculate current separation
        separation_vector = pos_j - pos_i
        current_distance = np.linalg.norm(separation_vector)
        
        if current_distance >= self.min_separation:
            return True  # Already separated
        
        # Calculate required correction
        required_distance = self.min_separation
        correction_magnitude = required_distance - current_distance
        
        # Unit vector pointing from ball_i to ball_j
        if current_distance > 1e-12:
            direction = separation_vector / current_distance
        else:
            # Balls are at same position, use random direction
            direction = np.array([1.0, 0.0, 0.0])
        
        # Apply correction by moving balls apart
        correction_i = -direction * (correction_magnitude / 2)
        correction_j = direction * (correction_magnitude / 2)
        
        # Apply corrections to ball motion events
        success_i = self._apply_ball_position_correction(physics_engine, ball_i, correction_i, correction_time)
        success_j = self._apply_ball_position_correction(physics_engine, ball_j, correction_j, correction_time)
        
        return success_i and success_j
    
    def _apply_ball_position_correction(self, physics_engine, ball_id: int, 
                                      correction: np.ndarray, correction_time: float) -> bool:
        """
        Apply position correction to a specific ball's motion event.
        """
        # Find active motion event
        if ball_id not in physics_engine._ball_motion_events:
            return False
        
        event = physics_engine._ball_motion_events[ball_id]
        
        # Check if correction time is within event
        if not (event.t <= correction_time <= event.t + event.T):
            return False
        
        # Apply correction to event's initial position
        if hasattr(event, 'r_0'):
            event.r_0 += correction
        elif hasattr(event, 'a') and len(event.a) > 0:
            event.a[0] += correction  # Position coefficients
        
        _logger.debug(f'Applied correction {correction} to ball {ball_id} at time {correction_time}')
        return True
    
    def _create_corrected_motion_events(self, physics_engine, ball_i: int, ball_j: int,
                                      new_pos_i: np.ndarray, new_pos_j: np.ndarray, 
                                      correction_time: float) -> bool:
        """
        Create new motion events with corrected positions.
        """
        from .events import BallRestEvent
        
        try:
            # Create rest events at corrected positions
            rest_event_i = BallRestEvent(correction_time, ball_i, new_pos_i)
            rest_event_j = BallRestEvent(correction_time, ball_j, new_pos_j)
            
            # Update physics engine state
            physics_engine._ball_motion_events[ball_i] = rest_event_i
            physics_engine._ball_motion_events[ball_j] = rest_event_j
            
            # Add to events list
            physics_engine.events.append(rest_event_i)
            physics_engine.events.append(rest_event_j)
            
            _logger.debug(f'Created corrected motion events for balls {ball_i}, {ball_j}')
            return True
            
        except Exception as e:
            _logger.error(f'Failed to create corrected motion events: {e}')
            return False


class CollisionCorrectionSystem:
    """
    Main collision correction system that integrates with physics engine.
    """
    
    def __init__(self, physics_engine):
        self.physics = physics_engine
        self.corrector = PenetrationCorrector(physics_engine.ball_radius)
        self.correction_history = []
        self.max_corrections_per_simulation = 10
        
    def check_and_correct_penetrations(self, current_time: float) -> bool:
        """
        Check for penetrations and correct them if found.
        
        Returns True if corrections were applied, False otherwise.
        """
        # Get current ball positions
        ball_positions = []
        active_balls = []
        
        for ball_id, event in self.physics._ball_motion_events.items():
            tau = current_time - event.t
            if 0 <= tau <= event.T:
                position = event.eval_position(tau)
                ball_positions.append(position)
                active_balls.append(ball_id)
        
        if len(ball_positions) < 2:
            return False
        
        # Check for penetrations
        penetration = self.corrector.detect_penetration(ball_positions)
        
        if penetration is None:
            return False
        
        # Apply correction
        pos_i, pos_j, depth = penetration
        ball_i = active_balls[pos_i]
        ball_j = active_balls[pos_j]
        
        _logger.warning(f'Penetration detected: balls {ball_i}, {ball_j}, depth={depth:.6f}mm at time {current_time}')
        
        # Check correction limit
        if len(self.correction_history) >= self.max_corrections_per_simulation:
            _logger.error('Maximum corrections reached, stopping simulation')
            return False
        
        # Apply correction
        success = self.corrector.correct_penetration(self.physics, current_time, ball_i, ball_j)
        
        if success:
            self.correction_history.append({
                'time': current_time,
                'balls': (ball_i, ball_j),
                'depth': depth
            })
        
        return success
    
    def get_correction_stats(self) -> Dict:
        """Get statistics about corrections applied."""
        return {
            'total_corrections': len(self.correction_history),
            'corrections': self.correction_history.copy()
        }


def create_collision_correction_system(physics_engine):
    """Factory function to create collision correction system."""
    return CollisionCorrectionSystem(physics_engine)
