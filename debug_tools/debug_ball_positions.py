#!/usr/bin/env python3
"""
Debug script to check ball positions and understand why balls go missing
"""
import numpy as np
from pool_physics.table import PoolTable
from pool_physics import PoolPhysics
from pool_physics.events import BallSlidingEvent

def debug_9_ball_break():
    """Debug what happens to balls in 9-ball break"""
    print("Debugging 9-ball break...")
    
    # Initialize physics
    physics = PoolPhysics()
    table = PoolTable()
    
    # Get rack positions
    initial_positions = table.calc_racked_positions(rack_type='9-ball')
    balls_on_table = list(range(10))  # 0-9
    
    # Reset physics with appropriate balls
    physics.reset(balls_on_table=balls_on_table, ball_positions=initial_positions)
    
    # Set up break shot parameters
    cue_ball_pos = initial_positions[0].copy()
    target_pos = initial_positions[1].copy()
    
    # Calculate direction vector from cue ball to target
    direction = target_pos - cue_ball_pos
    direction[1] = 0
    direction_norm = np.linalg.norm(direction)
    direction_unit = direction / direction_norm
    
    # Set up cue ball velocity (aimed at apex ball)
    v_0 = direction_unit * 12.0
    omega_0 = np.array([0.0, -5.0, 0.0])
    
    # Create break event
    break_event = BallSlidingEvent(
        t=0.0,
        i=0,
        r_0=cue_ball_pos,
        v_0=v_0,
        omega_0=omega_0
    )
    
    print("Initial positions:")
    for i, pos in enumerate(initial_positions[:10]):
        print(f"Ball {i}: {pos}")
    
    # Run simulation
    events = physics.add_event_sequence(break_event)
    
    # Get final positions
    final_time = 10.0
    final_positions = physics.eval_positions(final_time)
    
    print(f"\nFinal positions after {final_time}s:")
    for i in range(10):
        pos = final_positions[i]
        print(f"Ball {i}: {pos}")
        
        # Check various conditions
        is_on_table_old = pos[1] >= table.H - table.ball_radius  # Old logic
        is_at_origin = abs(pos[0]) < 1e-6 and abs(pos[2]) < 1e-6
        
        print(f"  - Y position: {pos[1]:.6f}")
        print(f"  - Table height: {table.H:.6f}")
        print(f"  - Ball radius: {table.ball_radius:.6f}")
        print(f"  - Threshold (H - R): {table.H - table.ball_radius:.6f}")
        print(f"  - On table (old logic): {is_on_table_old}")
        print(f"  - At origin: {is_at_origin}")
        print()
    
    # Count balls in different states
    on_table = 0
    below_threshold = 0
    at_origin = 0
    
    for i in range(10):
        pos = final_positions[i]
        if pos[1] >= table.H - table.ball_radius:
            on_table += 1
        if pos[1] < table.H - table.ball_radius:
            below_threshold += 1
        if abs(pos[0]) < 1e-6 and abs(pos[2]) < 1e-6:
            at_origin += 1
    
    print(f"Summary:")
    print(f"Balls on table (y >= {table.H - table.ball_radius:.6f}): {on_table}")
    print(f"Balls below threshold: {below_threshold}")
    print(f"Balls at origin: {at_origin}")
    print(f"Total events: {len(events)}")

if __name__ == "__main__":
    debug_9_ball_break()