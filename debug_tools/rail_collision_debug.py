#!/usr/bin/env python3
"""
Debug rail collision detection issues
"""

import sys
import os
import numpy as np
from numpy import dot, sqrt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pool_physics import PoolPhysics
from pool_physics.table import PoolTable


def debug_failing_ball(ball_id=3):
    """Debug why a specific ball is going off table"""
    
    # Set up same break shot
    physics = PoolPhysics()
    table = PoolTable()
    
    positions = table.calc_racked_positions('9-ball', spacing_mode='random', seed=42)
    physics.reset(ball_positions=positions, balls_on_table=list(range(10)))
    
    events = physics.strike_ball(0.0, 0, 
                               positions[0],
                               positions[0] + np.array([0, 0, physics.ball_radius]),
                               np.array([0.02, 0.0, -14.0]), 0.54)
    
    print(f"DEBUGGING BALL {ball_id} RAIL COLLISION FAILURE")
    print("=" * 60)
    
    # Find events for this ball
    ball_events = [e for e in events if hasattr(e, 'i') and e.i == ball_id]
    
    print(f"Ball {ball_id} events:")
    for i, event in enumerate(ball_events):
        print(f"  {i}: {type(event).__name__} at t={event.t:.6f}")
        if hasattr(event, 'T'):
            print(f"      Duration: {event.T:.6f}s")
    
    # Check the ball's motion at t=0.5s (when it should hit rail)
    test_time = 0.8  # When ball should be near rail
    
    print(f"\nAnalyzing ball motion at t={test_time}s:")
    
    # Find the active event at this time
    active_event = None
    for event in ball_events:
        if hasattr(event, 'T') and event.t <= test_time <= event.t + event.T:
            active_event = event
            break
    
    if active_event is None:
        print("❌ No active event found at this time")
        return
    
    print(f"Active event: {type(active_event).__name__}")
    
    # Get motion state
    tau = test_time - active_event.t
    pos = active_event.eval_position(tau)
    vel = active_event.eval_velocity(tau)
    
    print(f"Position: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
    print(f"Velocity: vx={vel[0]:.3f}, vy={vel[1]:.3f}, vz={vel[2]:.3f}")
    print(f"Speed: {np.linalg.norm(vel):.3f} m/s")
    
    # Check distance to table edges
    table_w, table_l = physics.table.W/2, physics.table.L/2
    
    print(f"\nTable boundaries: X=±{table_w:.3f}, Z=±{table_l:.3f}")
    print(f"Distance to rails:")
    print(f"  Right rail (+X): {table_w - pos[0]:.3f}m")
    print(f"  Left rail (-X):  {table_w + pos[0]:.3f}m") 
    print(f"  Top rail (+Z):   {table_l - pos[2]:.3f}m")
    print(f"  Bottom rail (-Z): {table_l + pos[2]:.3f}m")
    
    # Manual rail collision calculation
    print(f"\nMANUAL RAIL COLLISION CALCULATION:")
    
    # Check collision with right rail (most likely)
    if vel[0] > 0:  # Moving toward +X rail
        time_to_rail = (table_w - physics.ball_radius - pos[0]) / vel[0]
        print(f"Time to right rail: {time_to_rail:.3f}s")
        
        if time_to_rail > 0:
            collision_pos = pos + vel * time_to_rail
            print(f"Predicted collision position: x={collision_pos[0]:.3f}, z={collision_pos[2]:.3f}")
            
            if abs(collision_pos[2]) <= table_l:
                print("✅ Should collide with right rail")
            else:
                print("❌ Would miss rail - ball going to corner")
    
    # Test the actual collision detection function
    print(f"\nTESTING PHYSICS ENGINE COLLISION DETECTION:")
    
    # Check what the physics engine's segment collision detection returns
    segments = physics._segments
    
    for i_seg, (r_0, r_1, nor, tan) in enumerate(segments):
        # Test collision with this segment
        a0, a1, a2 = active_event._a
        A = dot(a2, nor)
        B = dot(a1, nor)
        C = dot((a0 - r_0), nor) - physics.ball_radius
        DD = B**2 - 4*A*C
        
        if i_seg < 4:  # Only check main rails
            seg_names = ["Bottom", "Right", "Top", "Left"]
            print(f"  {seg_names[i_seg]} rail (seg {i_seg}):")
            print(f"    A (accel·normal): {A:.6f}")
            print(f"    B (vel·normal): {B:.6f}") 
            print(f"    C (dist·normal): {C:.6f}")
            print(f"    Discriminant: {DD:.6f}")
            
            if abs(A) < 1e-15:
                print(f"    ❌ COLLISION SKIPPED: A too small ({A:.2e})")
            elif DD < 0:
                print(f"    ❌ No real solutions")
            else:
                D = sqrt(DD)
                tau_p = 0.5 * (-B + D) / A
                tau_n = 0.5 * (-B - D) / A
                print(f"    Collision times: {tau_n:.6f}, {tau_p:.6f}")
                
                # Check which one is valid
                for tau_test in [tau_n, tau_p]:
                    if 0 < tau_test < active_event.T:
                        r = active_event.eval_position(tau_test)
                        bounds_ok = (0 < dot(r - r_0, tan) < dot(r_1 - r_0, tan) and 
                                   0 < dot(r - r_0, nor))
                        print(f"      τ={tau_test:.6f}: bounds_ok={bounds_ok}")


if __name__ == "__main__":
    debug_failing_ball(3)  # Ball 3 is going off table
    print("\n" + "="*60)
    debug_failing_ball(9)  # Ball 9 is also going off table