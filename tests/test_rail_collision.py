#!/usr/bin/env python3
"""
Test script to reproduce and analyze rail collision detection bugs
"""
import numpy as np
from pool_physics import PoolPhysics
from pool_physics.events import BallSlidingEvent, RailCollisionEvent, SegmentCollisionEvent

def test_rail_collision_detection():
    """Test rail collision detection with a simple case"""
    
    # Initialize physics
    physics = PoolPhysics()
    table = physics.table
    
    print(f"Table dimensions: W={table.W:.3f}m, L={table.L:.3f}m")
    print(f"Ball radius: {physics.ball_radius:.4f}m")
    print(f"Expected boundary: X=±{table.W/2:.3f}m, Z=±{table.L/2:.3f}m")
    
    # Reset with single ball at center
    physics.reset(balls_on_table=[0])
    initial_pos = np.array([0.0, table.H + physics.ball_radius, 0.0])
    
    # Test case 1: Ball moving directly toward right wall
    print("\n=== TEST 1: Ball moving toward right wall ===")
    v_0 = np.array([1.0, 0.0, 0.0])  # Moving right at 1 m/s
    omega_0 = np.zeros(3)
    
    # Create motion event
    motion_event = BallSlidingEvent(
        t=0.0,
        i=0, 
        r_0=initial_pos,
        v_0=v_0,
        omega_0=omega_0
    )
    
    # Test rail collision detection directly
    rail_collision = physics._find_rail_collision(motion_event)
    
    if rail_collision:
        collision_time, e_i, seg_or_corner = rail_collision
        print(f"Rail collision found!")
        print(f"  Collision time: {collision_time:.4f}s")
        print(f"  Segment/corner: {seg_or_corner}")
        
        # Check position at collision time
        tau = collision_time - motion_event.t
        collision_pos = motion_event.eval_position(tau)
        print(f"  Collision position: ({collision_pos[0]:.4f}, {collision_pos[1]:.4f}, {collision_pos[2]:.4f})")
        
        # Check if this is reasonable
        expected_collision_x = table.W/2 - physics.ball_radius
        print(f"  Expected collision X: {expected_collision_x:.4f}")
        print(f"  Actual collision X: {collision_pos[0]:.4f}")
        print(f"  Difference: {abs(collision_pos[0] - expected_collision_x):.6f}")
    else:
        print("NO RAIL COLLISION DETECTED!")
        
        # Calculate where ball would be at expected collision time
        expected_time = (table.W/2 - physics.ball_radius) / v_0[0]
        expected_pos = motion_event.eval_position(expected_time)
        print(f"  Expected collision time: {expected_time:.4f}s") 
        print(f"  Position at expected time: ({expected_pos[0]:.4f}, {expected_pos[1]:.4f}, {expected_pos[2]:.4f})")
    
    # Test case 2: Ball moving at slight angle
    print("\n=== TEST 2: Ball moving at slight angle ===")
    v_0 = np.array([0.8, 0.0, 0.1])  # Moving mostly right, slightly forward
    
    motion_event2 = BallSlidingEvent(
        t=0.0,
        i=0,
        r_0=initial_pos, 
        v_0=v_0,
        omega_0=omega_0
    )
    
    rail_collision2 = physics._find_rail_collision(motion_event2)
    
    if rail_collision2:
        collision_time, e_i, seg_or_corner = rail_collision2
        print(f"Rail collision found!")
        print(f"  Collision time: {collision_time:.4f}s")
        print(f"  Segment/corner: {seg_or_corner}")
        
        tau = collision_time - motion_event2.t
        collision_pos = motion_event2.eval_position(tau)
        print(f"  Collision position: ({collision_pos[0]:.4f}, {collision_pos[1]:.4f}, {collision_pos[2]:.4f})")
    else:
        print("NO RAIL COLLISION DETECTED!")
        
    # Test segments setup
    print(f"\n=== SEGMENTS DEBUG INFO ===")
    print(f"Number of segments: {len(physics._segments)}")
    
    for i, (r_0, r_1, nor, tan) in enumerate(physics._segments[:6]):  # Show first 6
        print(f"Segment {i}:")
        print(f"  Start: ({r_0[0]:.3f}, {r_0[1]:.3f}, {r_0[2]:.3f})")
        print(f"  End: ({r_1[0]:.3f}, {r_1[1]:.3f}, {r_1[2]:.3f})")
        print(f"  Normal: ({nor[0]:.3f}, {nor[1]:.3f}, {nor[2]:.3f})")
        print(f"  Tangent: ({tan[0]:.3f}, {tan[1]:.3f}, {tan[2]:.3f})")
        
    # Test the collision math directly
    print(f"\n=== DIRECT COLLISION MATH TEST ===")
    
    # Use the first segment (should be a side wall)
    seg = physics._segments[0]
    r_0, r_1, nor, tan = seg
    
    print(f"Testing against segment 0")
    print(f"Normal vector: ({nor[0]:.3f}, {nor[1]:.3f}, {nor[2]:.3f})")
    
    # Test the problematic collision calculation
    a0, a1, a2 = motion_event._a
    A = np.dot(a2, nor)
    B = np.dot(a1, nor)
    C = np.dot((a0 - r_0), nor) - physics.ball_radius
    
    print(f"A (acceleration·normal): {A:.6f}")
    print(f"B (velocity·normal): {B:.6f}")  
    print(f"C (position offset): {C:.6f}")
    print(f"Discriminant: {B**2 - 4*A*C:.6f}")
    
    if abs(A) < 1e-10:
        print("*** A ≈ 0: Division by zero issue! ***")
        print("This explains why rail collision detection fails.")
        
        if abs(B) > 1e-10:
            # Linear case: solve Bt + C = 0
            tau_linear = -C / B
            print(f"Linear solution: tau = {tau_linear:.4f}s")
            if tau_linear > 0:
                pos_at_collision = motion_event.eval_position(tau_linear)
                print(f"Position at linear collision: ({pos_at_collision[0]:.4f}, {pos_at_collision[1]:.4f}, {pos_at_collision[2]:.4f})")

if __name__ == "__main__":
    test_rail_collision_detection()