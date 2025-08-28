"""
Adaptive time stepping system for improved collision detection accuracy.

This module implements variable time step sizing to provide high temporal
resolution during collision events while maintaining performance during
normal ball motion.
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Dict
from .events import BallMotionEvent, BallCollisionEvent
from .poly_solvers import f_find_collision_time as find_collision_time

_logger = logging.getLogger(__name__)


class AdaptiveTimeStepController:
    """
    Controls adaptive time stepping for physics simulation.
    
    Provides microsecond precision during collision events and millisecond
    precision during normal motion for optimal accuracy/performance balance.
    """
    
    def __init__(self, 
                 normal_dt: float = 1e-3,      # 1ms normal time step
                 collision_dt: float = 1e-6,   # 1μs collision time step
                 collision_window: float = 5e-3, # 5ms window around collisions
                 velocity_threshold: float = 0.1): # m/s threshold for high-speed detection
        
        self.normal_dt = normal_dt
        self.collision_dt = collision_dt
        self.collision_window = collision_window
        self.velocity_threshold = velocity_threshold
        
        # State tracking
        self.collision_events: List[Tuple[float, int, int]] = []
        self.high_speed_regions: List[Tuple[float, float]] = []
        
    def predict_collision_events(self, active_events: List[BallMotionEvent], 
                                current_time: float, lookahead: float = 0.1) -> List[Tuple[float, int, int]]:
        """
        Predict upcoming collision events within lookahead time.
        
        Returns list of (collision_time, ball_i, ball_j) tuples.
        """
        predicted_collisions = []
        
        for i, event_i in enumerate(active_events):
            for j, event_j in enumerate(active_events[i+1:], i+1):
                # Check temporal overlap
                t0 = max(event_i.t, event_j.t, current_time)
                t1 = min(event_i.t + event_i.T, event_j.t + event_j.T, current_time + lookahead)
                
                if t1 <= t0:
                    continue
                
                # Find collision time
                a_i = event_i.global_linear_motion_coeffs
                a_j = event_j.global_linear_motion_coeffs
                t_c = find_collision_time(a_i, a_j, event_i.ball_radius, t0, t1)
                
                if t_c is not None and current_time <= t_c <= current_time + lookahead:
                    predicted_collisions.append((t_c, event_i.i, event_j.i))
        
        return sorted(predicted_collisions)
    
    def detect_high_speed_regions(self, active_events: List[BallMotionEvent],
                                 current_time: float, lookahead: float = 0.1) -> List[Tuple[float, float]]:
        """
        Detect time regions where balls are moving at high speed.
        
        Returns list of (start_time, end_time) tuples for high-speed regions.
        """
        high_speed_regions = []
        
        for event in active_events:
            # Check if ball velocity exceeds threshold
            t_start = max(event.t, current_time)
            t_end = min(event.t + event.T, current_time + lookahead)
            
            if t_end <= t_start:
                continue
            
            # Sample velocity at multiple points
            sample_times = np.linspace(t_start, t_end, 10)
            for t in sample_times:
                tau = t - event.t
                velocity = event.eval_velocity(tau)
                speed = np.linalg.norm(velocity)
                
                if speed > self.velocity_threshold:
                    # Extend region around high-speed detection
                    region_start = max(t - self.collision_window/2, current_time)
                    region_end = min(t + self.collision_window/2, current_time + lookahead)
                    high_speed_regions.append((region_start, region_end))
                    break
        
        # Merge overlapping regions
        if high_speed_regions:
            high_speed_regions.sort()
            merged = [high_speed_regions[0]]
            
            for start, end in high_speed_regions[1:]:
                if start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            
            high_speed_regions = merged
        
        return high_speed_regions
    
    def generate_adaptive_time_points(self, current_time: float, end_time: float,
                                    active_events: List[BallMotionEvent]) -> np.ndarray:
        """
        Generate adaptive time points with high resolution near collisions.
        
        Returns array of time points with variable spacing.
        """
        # Predict collision events and high-speed regions
        lookahead = end_time - current_time
        collision_events = self.predict_collision_events(active_events, current_time, lookahead)
        high_speed_regions = self.detect_high_speed_regions(active_events, current_time, lookahead)
        
        # Create time point list
        time_points = []
        t = current_time
        
        while t < end_time:
            # Determine appropriate time step
            dt = self.normal_dt
            
            # Check if we're near a collision event
            for collision_time, _, _ in collision_events:
                if abs(t - collision_time) < self.collision_window:
                    dt = self.collision_dt
                    break
            
            # Check if we're in a high-speed region
            for region_start, region_end in high_speed_regions:
                if region_start <= t <= region_end:
                    dt = min(dt, self.collision_dt)
                    break
            
            time_points.append(t)
            t += dt
        
        # Ensure we include the end time
        if time_points[-1] < end_time:
            time_points.append(end_time)
        
        return np.array(time_points)
    
    def get_collision_window_points(self, collision_time: float, 
                                  window_size: float = None) -> np.ndarray:
        """
        Generate high-resolution time points around a specific collision.
        
        Returns dense time sampling around collision event.
        """
        if window_size is None:
            window_size = self.collision_window
        
        start_time = collision_time - window_size/2
        end_time = collision_time + window_size/2
        
        # Generate points with collision-level resolution
        num_points = int(window_size / self.collision_dt)
        return np.linspace(start_time, end_time, num_points)


class CollisionEventPredictor:
    """
    Predicts and tracks collision events for adaptive time stepping.
    """
    
    def __init__(self, prediction_horizon: float = 0.1):
        self.prediction_horizon = prediction_horizon
        self.predicted_events: Dict[Tuple[int, int], float] = {}
        
    def update_predictions(self, active_events: List[BallMotionEvent], 
                          current_time: float) -> List[Tuple[float, int, int]]:
        """
        Update collision predictions and return upcoming events.
        """
        self.predicted_events.clear()
        upcoming_collisions = []
        
        for i, event_i in enumerate(active_events):
            for j, event_j in enumerate(active_events[i+1:], i+1):
                collision_time = self._predict_collision(event_i, event_j, current_time)
                
                if collision_time is not None:
                    ball_pair = (min(event_i.i, event_j.i), max(event_i.i, event_j.i))
                    self.predicted_events[ball_pair] = collision_time
                    upcoming_collisions.append((collision_time, event_i.i, event_j.i))
        
        return sorted(upcoming_collisions)
    
    def _predict_collision(self, event_i: BallMotionEvent, event_j: BallMotionEvent,
                          current_time: float) -> Optional[float]:
        """
        Predict collision time between two ball events.
        """
        # Calculate temporal overlap
        t0 = max(event_i.t, event_j.t, current_time)
        t1 = min(event_i.t + event_i.T, event_j.t + event_j.T, 
                current_time + self.prediction_horizon)
        
        if t1 <= t0:
            return None
        
        # Use existing collision detection
        a_i = event_i.global_linear_motion_coeffs
        a_j = event_j.global_linear_motion_coeffs
        
        return find_collision_time(a_i, a_j, event_i.ball_radius, t0, t1)
    
    def is_collision_imminent(self, ball_i: int, ball_j: int, 
                            current_time: float, threshold: float = 0.01) -> bool:
        """
        Check if collision between two balls is imminent.
        """
        ball_pair = (min(ball_i, ball_j), max(ball_i, ball_j))
        
        if ball_pair not in self.predicted_events:
            return False
        
        collision_time = self.predicted_events[ball_pair]
        return collision_time - current_time <= threshold


def create_adaptive_time_controller(normal_dt: float = 1e-3, 
                                  collision_dt: float = 1e-6) -> AdaptiveTimeStepController:
    """Factory function to create adaptive time step controller."""
    return AdaptiveTimeStepController(normal_dt=normal_dt, collision_dt=collision_dt)
