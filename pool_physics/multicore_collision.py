"""
Multicore collision detection for pool physics engine.

This module implements parallel collision detection using data forks
to handle the time series constraints of the physics engine.
"""

import logging
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import List, Tuple, Optional, Any
import multiprocessing as mp
from functools import partial
import time

from .poly_solvers import f_find_collision_time as find_collision_time
from .adaptive_time_stepping import AdaptiveTimeStepController, CollisionEventPredictor
from .continuous_collision_detection import CCDMulticoreCollisionDetector
from .events import BallMotionEvent, BallCollisionEvent

_logger = logging.getLogger(__name__)


class MultiCoreCollisionDetector:
    """
    Parallel collision detection system that works with time series physics.
    
    Strategy:
    1. Fork ball motion data at specific time points
    2. Parallelize collision time finding across ball pairs
    3. Use adaptive time stepping for detailed collision analysis
    4. Merge results back into main physics timeline
    """
    
    def __init__(self, physics_engine, max_cores=None):
        self.physics = physics_engine
        self.max_cores = min(max_cores or cpu_count(), 8)
        self.ball_radius = physics_engine.ball_radius
        
        # Initialize adaptive time stepping components
        self.time_controller = AdaptiveTimeStepController(
            normal_dt=1e-3,      # 1ms normal resolution
            collision_dt=1e-6,   # 1μs collision resolution
            collision_window=5e-3, # 5ms window around collisions
            velocity_threshold=0.5  # m/s threshold for high-speed detection
        )
        self.collision_predictor = CollisionEventPredictor(prediction_horizon=0.1)
        
        # Initialize CCD detector
        self.ccd_detector = CCDMulticoreCollisionDetector(physics_engine, max_cores, ccd_tolerance=1e-9)
        self.use_ccd = False  # Can be enabled for zero-penetration guarantee
    
    def find_all_collisions_parallel(self, active_events: List[BallMotionEvent], 
                                   current_time: float) -> List[Tuple[float, int, int]]:
        """
        Find all potential ball-ball collisions in parallel with adaptive time stepping or CCD.
        
        Returns list of (collision_time, ball_i, ball_j) tuples.
        """
        if len(active_events) < 2:
            return []
        
        # Use CCD if enabled for zero-penetration guarantee
        if self.use_ccd:
            return self.ccd_detector.find_all_collisions_ccd(active_events, current_time)
        
        # Otherwise use adaptive time stepping approach
        # Update collision predictions for adaptive time stepping
        predicted_collisions = self.collision_predictor.update_predictions(active_events, current_time)
        _logger.debug(f'Predicted {len(predicted_collisions)} upcoming collisions')
        
        # Create all ball pair combinations with priority for predicted collisions
        ball_pairs = []
        priority_pairs = []
        
        for i, event_i in enumerate(active_events):
            for j, event_j in enumerate(active_events[i+1:], i+1):
                if self._should_check_collision(event_i, event_j):
                    pair_data = (event_i, event_j, i, j)
                    
                    # Check if this pair has a predicted collision
                    if self.collision_predictor.is_collision_imminent(event_i.i, event_j.i, current_time):
                        priority_pairs.append(pair_data)
                    else:
                        ball_pairs.append(pair_data)
        
        # Process priority pairs first (imminent collisions)
        all_pairs = priority_pairs + ball_pairs
        
        if not all_pairs:
            return []
        
        _logger.debug(f'Checking {len(all_pairs)} ball pairs ({len(priority_pairs)} priority) across {self.max_cores} cores')
        
        # For small numbers of pairs, use sequential to avoid multiprocessing overhead
        if len(all_pairs) <= 4:
            return self._process_collision_chunk_adaptive(all_pairs, current_time, active_events)
        
        # Split pairs across cores
        chunk_size = max(1, len(all_pairs) // self.max_cores)
        pair_chunks = [all_pairs[i:i + chunk_size] 
                      for i in range(0, len(all_pairs), chunk_size)]
        
        # Process chunks in parallel with timeout protection
        try:
            with Pool(self.max_cores) as pool:
                # Use partial to pass additional parameters
                process_func = partial(self._process_collision_chunk_adaptive_static, 
                                     current_time=current_time, 
                                     ball_radius=self.ball_radius)
                async_result = pool.map_async(process_func, pair_chunks)
                chunk_results = async_result.get(timeout=30)  # 30 second timeout
        except Exception as e:
            _logger.warning(f'Multicore collision detection failed: {e}, falling back to sequential')
            return self._process_collision_chunk_adaptive(all_pairs, current_time, active_events)
        
        # Flatten results
        all_collisions = []
        for chunk_result in chunk_results:
            all_collisions.extend(chunk_result)
        
        return sorted(all_collisions)  # Sort by collision time
    
    def enable_ccd(self, enable: bool = True):
        """Enable or disable Continuous Collision Detection."""
        self.use_ccd = enable
        _logger.info(f'CCD {"enabled" if enable else "disabled"}')
    
    def _should_check_collision(self, event_i: BallMotionEvent, 
                              event_j: BallMotionEvent) -> bool:
        """Check if two events could potentially collide."""
        # Basic temporal overlap check
        t0 = max(event_i.t, event_j.t)
        t1 = min(event_i.t + event_i.T, event_j.t + event_j.T)
        return t1 > t0
    
    def _process_collision_chunk(self, ball_pairs: List[Tuple]) -> List[Tuple[float, int, int]]:
        """Process a chunk of ball pairs for collision detection."""
        import os
        _logger.debug(f'Process {os.getpid()} checking {len(ball_pairs)} pairs')
        
        collisions = []
        for event_i, event_j, i, j in ball_pairs:
            collision_time = self._find_detailed_collision_time(event_i, event_j)
            if collision_time is not None:
                collisions.append((collision_time, i, j))
        
        return collisions
    
    def _process_collision_chunk_adaptive(self, ball_pairs: List[Tuple], 
                                        current_time: float, 
                                        active_events: List[BallMotionEvent]) -> List[Tuple[float, int, int]]:
        """Process collision chunk with adaptive time stepping."""
        import os
        _logger.debug(f'Process {os.getpid()} checking {len(ball_pairs)} pairs with adaptive stepping')
        
        collisions = []
        for event_i, event_j, i, j in ball_pairs:
            collision_time = self._find_adaptive_collision_time(event_i, event_j, current_time)
            if collision_time is not None:
                collisions.append((collision_time, i, j))
        
        return collisions
    
    @staticmethod
    def _process_collision_chunk_adaptive_static(ball_pairs: List[Tuple], 
                                               current_time: float, 
                                               ball_radius: float) -> List[Tuple[float, int, int]]:
        """Static method for multiprocessing with adaptive collision detection."""
        import os
        _logger.debug(f'Process {os.getpid()} checking {len(ball_pairs)} pairs (static adaptive)')
        
        collisions = []
        for event_i, event_j, i, j in ball_pairs:
            collision_time = MultiCoreCollisionDetector._find_adaptive_collision_time_static(
                event_i, event_j, current_time, ball_radius)
            if collision_time is not None:
                collisions.append((collision_time, i, j))
        
        return collisions
    
    def _find_detailed_collision_time(self, event_i: BallMotionEvent, 
                                    event_j: BallMotionEvent) -> Optional[float]:
        """
        Find collision time with same precision as sequential version.
        
        Uses data fork strategy but maintains compatibility with sequential behavior.
        """
        # Calculate temporal overlap
        t0 = max(event_i.t, event_j.t)
        t1 = min(event_i.t + event_i.T, event_j.t + event_j.T)
        
        if t1 <= t0:
            return None
        
        # Fork motion coefficients for independent calculation
        a_i = self._fork_motion_coeffs(event_i)
        a_j = self._fork_motion_coeffs(event_j)
        
        # Use same collision detection as sequential (no subdivision for consistency)
        return find_collision_time(a_i, a_j, self.ball_radius, t0, t1)
    
    def _find_adaptive_collision_time(self, event_i: BallMotionEvent, 
                                    event_j: BallMotionEvent, current_time: float) -> Optional[float]:
        """Find collision time using adaptive time stepping for higher precision."""
        # Calculate temporal overlap
        t0 = max(event_i.t, event_j.t, current_time)
        t1 = min(event_i.t + event_i.T, event_j.t + event_j.T)
        
        if t1 <= t0:
            return None
        
        # Generate adaptive time points for this collision search
        time_points = self.time_controller.generate_adaptive_time_points(
            t0, t1, [event_i, event_j])
        
        # Search for collision using adaptive time resolution
        return self._search_collision_adaptive_points(event_i, event_j, time_points)
    
    @staticmethod
    def _find_adaptive_collision_time_static(event_i: BallMotionEvent, 
                                           event_j: BallMotionEvent, 
                                           current_time: float, 
                                           ball_radius: float) -> Optional[float]:
        """Static version for multiprocessing."""
        # Calculate temporal overlap
        t0 = max(event_i.t, event_j.t, current_time)
        t1 = min(event_i.t + event_i.T, event_j.t + event_j.T)
        
        if t1 <= t0:
            return None
        
        # Use high-resolution search for static processing
        return MultiCoreCollisionDetector._high_resolution_collision_search(
            event_i, event_j, t0, t1, ball_radius)
    
    def _search_collision_adaptive_points(self, event_i: BallMotionEvent, 
                                        event_j: BallMotionEvent, 
                                        time_points: np.ndarray) -> Optional[float]:
        """Search for collision using adaptive time points."""
        a_i = self._fork_motion_coeffs(event_i)
        a_j = self._fork_motion_coeffs(event_j)
        
        # Search in segments defined by adaptive time points
        for i in range(len(time_points) - 1):
            t_start = time_points[i]
            t_end = time_points[i + 1]
            
            collision_time = find_collision_time(a_i, a_j, self.ball_radius, t_start, t_end)
            if collision_time is not None:
                return collision_time
        
        return None
    
    @staticmethod
    def _high_resolution_collision_search(event_i: BallMotionEvent, 
                                        event_j: BallMotionEvent,
                                        t0: float, t1: float, 
                                        ball_radius: float) -> Optional[float]:
        """High-resolution collision search for static processing."""
        a_i = event_i.global_linear_motion_coeffs.copy()
        a_j = event_j.global_linear_motion_coeffs.copy()
        
        # Use microsecond resolution for high-precision search
        dt = 1e-6  # 1 microsecond
        num_segments = max(1, int((t1 - t0) / dt))
        
        if num_segments > 10000:  # Limit to prevent excessive computation
            num_segments = 10000
            dt = (t1 - t0) / num_segments
        
        # Search in high-resolution segments
        for i in range(num_segments):
            seg_start = t0 + i * dt
            seg_end = min(t0 + (i + 1) * dt, t1)
            
            collision_time = find_collision_time(a_i, a_j, ball_radius, seg_start, seg_end)
            if collision_time is not None:
                return collision_time
        
        return None
    
    def _fork_motion_coeffs(self, event: BallMotionEvent) -> np.ndarray:
        """Create independent copy of motion coefficients."""
        return event.global_linear_motion_coeffs.copy()
    
    def _subdivided_collision_search(self, a_i: np.ndarray, a_j: np.ndarray,
                                   t0: float, t1: float, max_subdivisions: int = 3) -> Optional[float]:
        """
        Search for collision time using temporal subdivision.
        
        Recursively subdivides time interval for more precise collision detection.
        """
        # Try collision detection on full interval
        t_c = find_collision_time(a_i, a_j, self.ball_radius, t0, t1)
        
        if t_c is not None:
            return t_c
        
        # If no collision found and we can subdivide further
        if max_subdivisions > 0 and (t1 - t0) > 1e-6:  # 1 microsecond minimum
            # Split interval in half
            t_mid = (t0 + t1) / 2
            
            # Check first half
            t_c1 = self._subdivided_collision_search(a_i, a_j, t0, t_mid, max_subdivisions - 1)
            if t_c1 is not None:
                return t_c1
            
            # Check second half
            t_c2 = self._subdivided_collision_search(a_i, a_j, t_mid, t1, max_subdivisions - 1)
            if t_c2 is not None:
                return t_c2
        
        return None


def create_multicore_collision_detector(physics_engine, max_cores=None):
    """Factory function to create multicore collision detector."""
    return MultiCoreCollisionDetector(physics_engine, max_cores)


# Integration functions for existing physics engine
def find_next_collision_parallel(physics_engine, active_events: List[BallMotionEvent],
                                current_time: float) -> Optional[Tuple[float, BallCollisionEvent]]:
    """
    Find the next collision using parallel processing.
    
    This function integrates with the existing physics engine event loop.
    """
    detector = MultiCoreCollisionDetector(physics_engine)
    collisions = detector.find_all_collisions_parallel(active_events, current_time)
    
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
