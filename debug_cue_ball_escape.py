#!/usr/bin/env python3
"""
Debug why cue ball escapes despite rail collisions
"""
import numpy as np
from pool_physics.table import PoolTable
from pool_physics import PoolPhysics
from pool_physics.events import BallSlidingEvent

def debug_cue_ball_path():
    """Track cue ball through segment collisions"""
    
    # Initialize physics
    physics = PoolPhysics()
    table = PoolTable()
    
    # Get 8-ball rack positions
    initial_positions = table.calc_racked_positions(rack_type='8-ball')
    balls_on_table = list(range(16))  # 0-15
    
    # Reset physics with all balls
    physics.reset(balls_on_table=balls_on_table, ball_positions=initial_positions)
    
    # Set up break shot
    cue_ball_pos = initial_positions[0].copy()
    target_pos = initial_positions[1].copy()
    
    direction = target_pos - cue_ball_pos
    direction[1] = 0
    direction_norm = np.linalg.norm(direction)
    direction_unit = direction / direction_norm
    
    v_0 = direction_unit * 10.0
    omega_0 = np.array([0.0, -5.0, 0.0])
    
    break_event = BallSlidingEvent(
        t=0.0,
        i=0,
        r_0=cue_ball_pos,
        v_0=v_0,
        omega_0=omega_0
    )
    
    # Run simulation
    events = physics.add_event_sequence(break_event)
    
    print(f"Table boundaries: X=±{table.W/2:.3f}m, Z=±{table.L/2:.3f}m")
    print(f"Ball radius: {table.ball_radius:.4f}m")
    print(f"Safe boundaries (accounting for ball radius): X=±{table.W/2 - table.ball_radius:.3f}m, Z=±{table.L/2 - table.ball_radius:.3f}m")
    
    # Track cue ball events specifically
    cue_ball_events = [e for e in events if hasattr(e, 'i') and e.i == 0]
    segment_events = [e for e in events if type(e).__name__ == 'SegmentCollisionEvent']
    cue_segment_events = [e for e in segment_events if hasattr(e, 'e_i') and hasattr(e.e_i, 'i') and e.e_i.i == 0]
    
    print(f"\nCue ball (ball 0) events: {len(cue_ball_events)}")
    print(f"Total segment collisions: {len(segment_events)}")
    print(f"Cue ball segment collisions: {len(cue_segment_events)}")
    
    # Check cue ball position at segment collision times
    print(f"\nCue ball segment collision details:")
    for i, event in enumerate(cue_segment_events):
        t = event.t
        pos_before = physics.eval_positions(t - 0.001)[0]  # Just before collision
        pos_after = physics.eval_positions(t + 0.001)[0]   # Just after collision
        
        print(f"  Collision {i+1} at t={t:.4f}s:")
        print(f"    Before: ({pos_before[0]:.4f}, {pos_before[1]:.4f}, {pos_before[2]:.4f})")
        print(f"    After:  ({pos_after[0]:.4f}, {pos_after[1]:.4f}, {pos_after[2]:.4f})")
        
        # Check if positions are within bounds
        x_before, z_before = pos_before[0], pos_before[2]
        x_after, z_after = pos_after[0], pos_after[2]
        
        boundary_x = table.W/2 - table.ball_radius
        boundary_z = table.L/2 - table.ball_radius
        
        if abs(x_before) > boundary_x or abs(z_before) > boundary_z:
            print(f"    *** POSITION BEFORE COLLISION IS OUT OF BOUNDS! ***")
        if abs(x_after) > boundary_x or abs(z_after) > boundary_z:
            print(f"    *** POSITION AFTER COLLISION IS OUT OF BOUNDS! ***")
    
    # Check final position
    final_pos = physics.eval_positions(10.0)[0]
    print(f"\nFinal cue ball position at t=10s: ({final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f})")
    
    # Find when cue ball first goes out of bounds
    for t in np.arange(0, 10, 0.1):
        pos = physics.eval_positions(t)[0]
        x, z = pos[0], pos[2]
        boundary_x = table.W/2 - table.ball_radius
        boundary_z = table.L/2 - table.ball_radius
        
        if abs(x) > boundary_x or abs(z) > boundary_z:
            print(f"\nFirst boundary violation at t={t:.1f}s: ({x:.4f}, {z:.4f})")
            break
    
    return events

if __name__ == "__main__":
    debug_cue_ball_path()