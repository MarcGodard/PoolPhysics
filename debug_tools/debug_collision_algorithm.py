#!/usr/bin/env python3
"""
Debug the collision detection algorithm step by step
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pool_physics import PoolPhysics
from pool_physics.table import PoolTable
from pool_physics.events import BallSlidingEvent

def debug_collision_detection():
    """Debug the collision detection algorithm step by step"""
    
    table = PoolTable()
    physics = PoolPhysics(table=table)
    
    print("COLLISION DETECTION ALGORITHM DEBUG")
    print("=" * 60)
    
    # Set up ball moving toward +X rail
    initial_pos = np.array([0.0, table.H + physics.ball_radius, 0.0])
    initial_velocity = np.array([2.0, 0.0, 0.0])
    
    print(f"Ball position: {initial_pos}")
    print(f"Ball velocity: {initial_velocity}")
    
    # Create sliding event
    roll_event = BallSlidingEvent(
        t=0.0, i=0, r_0=initial_pos,
        v_0=initial_velocity, omega_0=np.array([0.0, 0.0, 0.0])
    )
    
    print(f"\nEvent created: {type(roll_event).__name__}")
    print(f"Event time: {roll_event.t}")
    print(f"Ball index: {roll_event.i}")
    
    # Reset physics and test collision detection manually
    physics.reset(balls_on_table=[0], ball_positions=np.array([initial_pos]))
    
    # Test the collision detection method directly
    print(f"\nTesting collision detection for each segment:")
    print("-" * 60)
    
    tau_min = float('inf')
    seg_min = None
    
    for i_seg, (r_0, r_1, nor, tan) in enumerate(physics._segments):
        # Check if this is a right wall segment (normal points -X)
        if abs(nor[0]) > 0.9 and nor[0] < 0:
            print(f"\nSegment {i_seg}: RIGHT WALL")
            print(f"  Start: {r_0}")
            print(f"  End: {r_1}")
            print(f"  Normal: {nor}")
            print(f"  Tangent: {tan}")
            
            # Call the collision detection method
            try:
                tau_n, tau_p = physics._find_segment_collision_time(roll_event, r_0, r_1, nor, tan)
                print(f"  Collision times: tau_n={tau_n:.6f}, tau_p={tau_p:.6f}")
                
                tau_n, tau_p = min(tau_n, tau_p), max(tau_n, tau_p)
                
                if 0 < tau_n < tau_min:
                    r = roll_event.eval_position(tau_n)
                    print(f"  Testing tau_n={tau_n:.6f}")
                    print(f"  Position at tau_n: {r}")
                    
                    # Check bounds
                    dot_tan = np.dot(r - r_0, tan)
                    dot_tan_max = np.dot(r_1 - r_0, tan)
                    dot_nor = np.dot(r - r_0, nor)
                    
                    print(f"  dot(r-r_0, tan) = {dot_tan:.6f}")
                    print(f"  dot(r_1-r_0, tan) = {dot_tan_max:.6f}")
                    print(f"  dot(r-r_0, nor) = {dot_nor:.6f}")
                    print(f"  Bounds check: 0 < {dot_tan:.3f} < {dot_tan_max:.3f} and 0 < {dot_nor:.3f}")
                    
                    if 0 < dot_tan < dot_tan_max and 0 < dot_nor:
                        print(f"  ✅ COLLISION DETECTED at tau_n={tau_n:.6f}")
                        tau_min = tau_n
                        seg_min = i_seg
                    else:
                        print(f"  ❌ Bounds check failed")
                        
                if 0 < tau_p < tau_min:
                    r = roll_event.eval_position(tau_p)
                    print(f"  Testing tau_p={tau_p:.6f}")
                    print(f"  Position at tau_p: {r}")
                    
                    # Check bounds
                    dot_tan = np.dot(r - r_0, tan)
                    dot_tan_max = np.dot(r_1 - r_0, tan)
                    dot_nor = np.dot(r - r_0, nor)
                    
                    print(f"  dot(r-r_0, tan) = {dot_tan:.6f}")
                    print(f"  dot(r_1-r_0, tan) = {dot_tan_max:.6f}")
                    print(f"  dot(r-r_0, nor) = {dot_nor:.6f}")
                    print(f"  Bounds check: 0 < {dot_tan:.3f} < {dot_tan_max:.3f} and 0 < {dot_nor:.3f}")
                    
                    if 0 < dot_tan < dot_tan_max and 0 < dot_nor:
                        print(f"  ✅ COLLISION DETECTED at tau_p={tau_p:.6f}")
                        tau_min = tau_p
                        seg_min = i_seg
                    else:
                        print(f"  ❌ Bounds check failed")
                        
            except Exception as e:
                print(f"  ERROR in collision detection: {e}")
    
    print(f"\n" + "=" * 60)
    print("FINAL RESULT:")
    if seg_min is not None:
        print(f"✅ Collision detected with segment {seg_min} at time {tau_min:.6f}")
    else:
        print(f"❌ No collision detected")
    
    # Now test the full event sequence
    print(f"\n" + "=" * 60)
    print("FULL EVENT SEQUENCE TEST:")
    events = physics.add_event_sequence(roll_event)
    rail_collisions = [e for e in events if 'Segment' in type(e).__name__]
    print(f"Events generated: {len(events)}")
    print(f"Rail collisions: {len(rail_collisions)}")
    
    for i, event in enumerate(events):
        print(f"  Event {i}: {type(event).__name__} at t={event.t:.6f}")

if __name__ == "__main__":
    debug_collision_detection()
