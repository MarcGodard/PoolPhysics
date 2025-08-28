"""
Continuous Collision Detection (CCD) for pool physics engine.

This module implements industry-standard CCD algorithms to prevent ball
penetrations entirely by detecting collisions during motion rather than
at discrete time points.
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Dict
import math
from .events import BallMotionEvent, BallCollisionEvent

_logger = logging.getLogger(__name__)


class SweptSphere:
    """
    Represents a sphere moving through space over a time interval.
    Used for continuous collision detection between moving balls.
    """
    
    def __init__(self, center_start: np.ndarray, center_end: np.ndarray, radius: float):
        self.center_start = np.array(center_start)
        self.center_end = np.array(center_end)
        self.radius = radius
        self.velocity = center_end - center_start
        
    def position_at_time(self, t: float) -> np.ndarray:
        """Get sphere center position at normalized time t ∈ [0,1]."""
        return self.center_start + t * self.velocity
    
    def distance_at_time(self, other: 'SweptSphere', t: float) -> float:
        """Calculate distance between sphere centers at time t."""
        pos1 = self.position_at_time(t)
        pos2 = other.position_at_time(t)
        return np.linalg.norm(pos2 - pos1)
    
    def collision_radius(self, other: 'SweptSphere') -> float:
        """Combined collision radius for two spheres."""
        return self.radius + other.radius


class ContinuousCollisionDetector:
    """
    Implements continuous collision detection using swept sphere algorithm
    with conservative advancement and binary search refinement.
    """
    
    def __init__(self, tolerance: float = 1e-9, max_iterations: int = 50):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        
    def find_collision_time(self, event_i: BallMotionEvent, event_j: BallMotionEvent,
                          t_start: float, t_end: float) -> Optional[float]:
        """
        Find exact collision time using continuous collision detection.
        
        Returns collision time or None if no collision occurs.
        """
        # Create swept spheres for the time interval
        pos_i_start = event_i.eval_position(t_start - event_i.t)
        pos_i_end = event_i.eval_position(t_end - event_i.t)
        pos_j_start = event_j.eval_position(t_start - event_j.t)
        pos_j_end = event_j.eval_position(t_end - event_j.t)
        
        sphere_i = SweptSphere(pos_i_start, pos_i_end, event_i.ball_radius)
        sphere_j = SweptSphere(pos_j_start, pos_j_end, event_j.ball_radius)
        
        # Check if collision occurs during interval
        collision_time = self._swept_sphere_collision(sphere_i, sphere_j, t_start, t_end)
        
        if collision_time is not None:
            # Validate collision with velocity check (balls must be approaching)
            if self._validate_collision_direction(event_i, event_j, collision_time):
                return collision_time
        
        return None
    
    def _swept_sphere_collision(self, sphere_i: SweptSphere, sphere_j: SweptSphere,
                              t_start: float, t_end: float) -> Optional[float]:
        """
        Find collision time between two swept spheres using conservative advancement.
        """
        # Check initial separation
        initial_distance = sphere_i.distance_at_time(sphere_j, 0.0)
        collision_radius = sphere_i.collision_radius(sphere_j)
        
        if initial_distance <= collision_radius:
            return t_start  # Already in contact
        
        # Check final separation
        final_distance = sphere_i.distance_at_time(sphere_j, 1.0)
        if final_distance > collision_radius:
            # Check if spheres get closer during motion
            if not self._spheres_approach(sphere_i, sphere_j):
                return None
        
        # Use conservative advancement with binary search
        return self._conservative_advancement(sphere_i, sphere_j, t_start, t_end)
    
    def _conservative_advancement(self, sphere_i: SweptSphere, sphere_j: SweptSphere,
                                t_start: float, t_end: float) -> Optional[float]:
        """
        Conservative advancement algorithm with binary search refinement.
        """
        dt = t_end - t_start
        collision_radius = sphere_i.collision_radius(sphere_j)
        
        # Binary search for collision time
        t_low = 0.0
        t_high = 1.0
        
        for iteration in range(self.max_iterations):
            t_mid = (t_low + t_high) / 2.0
            distance = sphere_i.distance_at_time(sphere_j, t_mid)
            
            if abs(distance - collision_radius) < self.tolerance:
                return t_start + t_mid * dt
            
            if distance > collision_radius:
                t_low = t_mid
            else:
                t_high = t_mid
            
            # Check convergence
            if (t_high - t_low) * dt < self.tolerance:
                return t_start + t_mid * dt
        
        # If no convergence, check if collision occurs
        final_distance = sphere_i.distance_at_time(sphere_j, t_high)
        if final_distance <= collision_radius + self.tolerance:
            return t_start + t_high * dt
        
        return None
    
    def _spheres_approach(self, sphere_i: SweptSphere, sphere_j: SweptSphere) -> bool:
        """Check if spheres are approaching each other."""
        # Calculate relative velocity
        rel_velocity = sphere_j.velocity - sphere_i.velocity
        rel_position = sphere_j.center_start - sphere_i.center_start
        
        # Spheres approach if relative velocity points toward relative position
        return np.dot(rel_velocity, rel_position) < 0
    
    def _validate_collision_direction(self, event_i: BallMotionEvent, event_j: BallMotionEvent,
                                    collision_time: float) -> bool:
        """Validate that balls are approaching at collision time."""
        # Get velocities at collision time
        tau_i = collision_time - event_i.t
        tau_j = collision_time - event_j.t
        
        v_i = event_i.eval_velocity(tau_i)
        v_j = event_j.eval_velocity(tau_j)
        
        # Get positions at collision time
        r_i = event_i.eval_position(tau_i)
        r_j = event_j.eval_position(tau_j)
        
        # Relative velocity and position
        v_rel = v_j - v_i
        r_rel = r_j - r_i
        
        # Balls must be approaching (relative velocity toward relative position)
        return np.dot(v_rel, r_rel) < 0


class CCDMulticoreCollisionDetector:
    """
    Multicore collision detection using Continuous Collision Detection.
    Combines CCD algorithms with parallel processing for performance.
    """
    
    def __init__(self, physics_engine, max_cores=None, ccd_tolerance=1e-9):
        from multiprocessing import cpu_count
        self.physics = physics_engine
        self.max_cores = min(max_cores or cpu_count(), 8)
        self.ball_radius = physics_engine.ball_radius
        self.ccd_detector = ContinuousCollisionDetector(tolerance=ccd_tolerance)
        
    def find_all_collisions_ccd(self, active_events: List[BallMotionEvent], 
                               current_time: float, lookahead: float = 0.1) -> List[Tuple[float, int, int]]:
        """
        Find all collisions using CCD with multicore processing.
        
        Returns list of (collision_time, ball_i, ball_j) tuples.
        """
        if len(active_events) < 2:
            return []
        
        # Create ball pair combinations
        ball_pairs = []
        for i, event_i in enumerate(active_events):
            for j, event_j in enumerate(active_events[i+1:], i+1):
                if self._should_check_collision_ccd(event_i, event_j, current_time, lookahead):
                    ball_pairs.append((event_i, event_j, i, j))
        
        if not ball_pairs:
            return []
        
        _logger.debug(f'CCD checking {len(ball_pairs)} ball pairs across {self.max_cores} cores')
        
        # For small numbers of pairs, use sequential processing
        if len(ball_pairs) <= 4:
            return self._process_ccd_chunk(ball_pairs, current_time, lookahead)
        
        # Process in parallel
        from multiprocessing import Pool
        from functools import partial
        
        chunk_size = max(1, len(ball_pairs) // self.max_cores)
        pair_chunks = [ball_pairs[i:i + chunk_size] 
                      for i in range(0, len(ball_pairs), chunk_size)]
        
        try:
            with Pool(self.max_cores) as pool:
                process_func = partial(self._process_ccd_chunk_static, 
                                     current_time=current_time, 
                                     lookahead=lookahead,
                                     ball_radius=self.ball_radius,
                                     tolerance=self.ccd_detector.tolerance)
                chunk_results = pool.map(process_func, pair_chunks)
        except Exception as e:
            _logger.warning(f'CCD multicore processing failed: {e}, falling back to sequential')
            return self._process_ccd_chunk(ball_pairs, current_time, lookahead)
        
        # Flatten and sort results
        all_collisions = []
        for chunk_result in chunk_results:
            all_collisions.extend(chunk_result)
        
        return sorted(all_collisions)
    
    def _should_check_collision_ccd(self, event_i: BallMotionEvent, event_j: BallMotionEvent,
                                   current_time: float, lookahead: float) -> bool:
        """Check if two events should be tested for CCD collision."""
        # Temporal overlap check
        t0 = max(event_i.t, event_j.t, current_time)
        t1 = min(event_i.t + event_i.T, event_j.t + event_j.T, current_time + lookahead)
        
        if t1 <= t0:
            return False
        
        # Quick distance check - skip if balls are very far apart
        pos_i = event_i.eval_position(t0 - event_i.t)
        pos_j = event_j.eval_position(t0 - event_j.t)
        distance = np.linalg.norm(pos_j - pos_i)
        
        # Skip if distance is much larger than possible collision distance
        max_speed = 10.0  # m/s reasonable maximum ball speed
        max_travel = max_speed * (t1 - t0)
        collision_radius = 2 * self.ball_radius
        
        return distance <= collision_radius + max_travel
    
    def _process_ccd_chunk(self, ball_pairs: List[Tuple], 
                          current_time: float, lookahead: float) -> List[Tuple[float, int, int]]:
        """Process collision chunk using CCD."""
        import os
        _logger.debug(f'Process {os.getpid()} CCD checking {len(ball_pairs)} pairs')
        
        collisions = []
        for event_i, event_j, i, j in ball_pairs:
            t_start = max(event_i.t, event_j.t, current_time)
            t_end = min(event_i.t + event_i.T, event_j.t + event_j.T, current_time + lookahead)
            
            collision_time = self.ccd_detector.find_collision_time(event_i, event_j, t_start, t_end)
            if collision_time is not None:
                collisions.append((collision_time, i, j))
        
        return collisions
    
    @staticmethod
    def _process_ccd_chunk_static(ball_pairs: List[Tuple], current_time: float, 
                                 lookahead: float, ball_radius: float, 
                                 tolerance: float) -> List[Tuple[float, int, int]]:
        """Static method for multiprocessing CCD."""
        import os
        _logger.debug(f'Process {os.getpid()} static CCD checking {len(ball_pairs)} pairs')
        
        detector = ContinuousCollisionDetector(tolerance=tolerance)
        collisions = []
        
        for event_i, event_j, i, j in ball_pairs:
            t_start = max(event_i.t, event_j.t, current_time)
            t_end = min(event_i.t + event_i.T, event_j.t + event_j.T, current_time + lookahead)
            
            collision_time = detector.find_collision_time(event_i, event_j, t_start, t_end)
            if collision_time is not None:
                collisions.append((collision_time, i, j))
        
        return collisions


def create_ccd_collision_detector(physics_engine, max_cores=None, tolerance=1e-9):
    """Factory function to create CCD collision detector."""
    return CCDMulticoreCollisionDetector(physics_engine, max_cores, tolerance)


# Integration functions for existing physics engine
def find_next_collision_ccd(physics_engine, active_events: List[BallMotionEvent],
                           current_time: float) -> Optional[Tuple[float, BallCollisionEvent]]:
    """
    Find the next collision using CCD algorithm.
    
    This function integrates with the existing physics engine event loop.
    """
    detector = CCDMulticoreCollisionDetector(physics_engine)
    collisions = detector.find_all_collisions_ccd(active_events, current_time)
    
    if not collisions:
        return None
    
    # Return earliest collision
    earliest_time, ball_i, ball_j = collisions[0]
    
    # Create collision event using existing physics engine logic
    event_i = active_events[ball_i]
    event_j = active_events[ball_j]
    
    # Use the physics engine's collision model
    collision_model = physics_engine.ball_collision_model
    collision_event = collision_model(earliest_time, event_i, event_j)
    
    return earliest_time, collision_event
