#!/usr/bin/env python3
"""
Debug rail bounce test to understand why no rail collisions occur
"""
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pool_physics.table import PoolTable
from pool_physics import PoolPhysics
from pool_physics.events import BallSlidingEvent

def debug_rail_bounce():
    """Debug why rail bounce test isn't working"""
    
    print("DEBUGGING RAIL BOUNCE TEST")
    print("=" * 50)
    
    # Initialize physics and table
    physics = PoolPhysics()
    table = PoolTable()
    
    # Print table dimensions
    print(f"Table dimensions:")
    print(f"  Width (W): {table.W:.3f} m")
    print(f"  Length (L): {table.L:.3f} m") 
    print(f"  Height (H): {table.H:.3f} m")
    print(f"  Ball radius: {table.ball_radius:.3f} m")
    print(f"  Rail positions: ±{table.W/2:.3f} m (X), ±{table.L/2:.3f} m (Z)")
    
    # Test different starting positions and velocities
    test_cases = [
        {
            'name': 'Original (W/3 toward +X)',
            'pos': np.array([table.W/3, table.H, 0.0]),
            'vel': np.array([3.0, 0.0, 0.0])
        },
        {
            'name': 'Center toward +X rail',
            'pos': np.array([0.0, table.H, 0.0]),
            'vel': np.array([3.0, 0.0, 0.0])
        },
        {
            'name': 'Near rail toward +X',
            'pos': np.array([table.W/2 - 0.2, table.H, 0.0]),
            'vel': np.array([2.0, 0.0, 0.0])
        },
        {
            'name': 'Center toward +Z rail',
            'pos': np.array([0.0, table.H, 0.0]),
            'vel': np.array([0.0, 0.0, 3.0])
        },
        {
            'name': 'Near +Z rail',
            'pos': np.array([0.0, table.H, table.L/2 - 0.2]),
            'vel': np.array([0.0, 0.0, 2.0])
        }
    ]
    
    for i, test in enumerate(test_cases):
        print(f"\nTest {i+1}: {test['name']}")
        print("-" * 40)
        
        # Reset physics
        physics.reset(balls_on_table=[0], ball_positions=np.array([test['pos']]))
        
        print(f"  Start position: [{test['pos'][0]:.3f}, {test['pos'][1]:.3f}, {test['pos'][2]:.3f}]")
        print(f"  Initial velocity: [{test['vel'][0]:.3f}, {test['vel'][1]:.3f}, {test['vel'][2]:.3f}]")
        
        # Calculate expected collision point
        if test['vel'][0] != 0:  # Moving in X direction
            rail_x = table.W/2 if test['vel'][0] > 0 else -table.W/2
            time_to_rail = (rail_x - test['pos'][0]) / test['vel'][0]
            expected_collision = test['pos'] + test['vel'] * time_to_rail
            print(f"  Expected rail hit at: [{expected_collision[0]:.3f}, {expected_collision[1]:.3f}, {expected_collision[2]:.3f}]")
            print(f"  Time to rail: {time_to_rail:.3f}s")
        elif test['vel'][2] != 0:  # Moving in Z direction
            rail_z = table.L/2 if test['vel'][2] > 0 else -table.L/2
            time_to_rail = (rail_z - test['pos'][2]) / test['vel'][2]
            expected_collision = test['pos'] + test['vel'] * time_to_rail
            print(f"  Expected rail hit at: [{expected_collision[0]:.3f}, {expected_collision[1]:.3f}, {expected_collision[2]:.3f}]")
            print(f"  Time to rail: {time_to_rail:.3f}s")
        
        # Create sliding event
        slide_event = BallSlidingEvent(
            t=0.0, i=0, r_0=test['pos'],
            v_0=test['vel'], omega_0=np.array([0.0, 0.0, 0.0])
        )
        
        # Run simulation
        events = physics.add_event_sequence(slide_event)
        
        # Analyze events
        print(f"  Total events: {len(events)}")
        
        event_types = {}
        for event in events:
            event_type = type(event).__name__
            if event_type not in event_types:
                event_types[event_type] = []
            event_types[event_type].append(event)
        
        for event_type, event_list in event_types.items():
            print(f"    {event_type}: {len(event_list)}")
            if 'Segment' in event_type or 'Rail' in event_type:
                for j, event in enumerate(event_list[:3]):
                    print(f"      Event {j}: t={event.t:.4f}s")
        
        # Check ball positions at various times
        time_points = [0.0, 0.1, 0.2, 0.5, 1.0]
        print(f"  Ball trajectory:")
        for t in time_points:
            try:
                pos = physics.eval_positions(t)[0]
                vel = physics.eval_velocities(t)[0]
                print(f"    t={t:.1f}s: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], vel=[{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}]")
            except:
                print(f"    t={t:.1f}s: simulation ended")
                break

if __name__ == "__main__":
    debug_rail_bounce()
