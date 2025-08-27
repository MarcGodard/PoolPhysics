#!/usr/bin/env python3
"""
Analyze the table segments to understand why rail collision detection fails
"""
import numpy as np
from pool_physics import PoolPhysics
from pool_physics.table import PoolTable

def analyze_table_segments():
    """Analyze how table segments are constructed"""
    
    physics = PoolPhysics()
    table = physics.table
    
    print(f"Table dimensions: W={table.W:.3f}m, L={table.L:.3f}m")
    print(f"Table boundaries should be: X=±{table.W/2:.3f}m, Z=±{table.L/2:.3f}m")
    
    # Show the raw corner data
    print(f"\n=== RAW TABLE CORNERS ===")
    print(f"Number of corners: {len(table._corners)}")
    
    for i, corner in enumerate(table._corners):
        print(f"Corner {i:2d}: ({corner[0]:7.3f}, {corner[1]:7.3f})")
    
    # Show the 3D corners used in physics
    print(f"\n=== 3D PHYSICS CORNERS ===")  
    print(f"Number of corners: {len(physics._corners)}")
    
    for i, corner in enumerate(physics._corners):
        print(f"Corner {i:2d}: ({corner[0]:7.3f}, {corner[1]:7.3f}, {corner[2]:7.3f})")
    
    # Show the segments
    print(f"\n=== TABLE SEGMENTS ===")
    print(f"Number of segments: {len(physics._segments)}")
    
    segment_indices = (0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22)  # From source code
    print(f"Segment indices used: {segment_indices}")
    
    for i, (r_0, r_1, nor, tan) in enumerate(physics._segments):
        length = np.linalg.norm(r_1 - r_0)
        print(f"Segment {i:2d}:")
        print(f"  Start: ({r_0[0]:7.3f}, {r_0[1]:7.3f}, {r_0[2]:7.3f})")
        print(f"  End:   ({r_1[0]:7.3f}, {r_1[1]:7.3f}, {r_1[2]:7.3f})")
        print(f"  Normal:  ({nor[0]:6.3f}, {nor[1]:6.3f}, {nor[2]:6.3f})")
        print(f"  Tangent: ({tan[0]:6.3f}, {tan[1]:6.3f}, {tan[2]:6.3f})")
        print(f"  Length: {length:.3f}m")
        
        # Determine what kind of wall this represents
        wall_type = "unknown"
        if abs(nor[0]) > 0.9:  # X-normal (left/right wall)
            wall_type = "left wall" if nor[0] < 0 else "right wall"
        elif abs(nor[2]) > 0.9:  # Z-normal (top/bottom wall)
            wall_type = "bottom wall" if nor[2] < 0 else "top wall"
        elif abs(nor[0]) > 0.3 and abs(nor[2]) > 0.3:  # Diagonal (corner)
            wall_type = "corner/pocket segment"
        
        print(f"  Type: {wall_type}")
        print()
    
    # Check if we have proper wall segments
    print(f"=== WALL SEGMENT ANALYSIS ===")
    
    has_left_wall = any(abs(nor[0] + 1) < 0.1 for _, _, nor, _ in physics._segments)
    has_right_wall = any(abs(nor[0] - 1) < 0.1 for _, _, nor, _ in physics._segments)  
    has_bottom_wall = any(abs(nor[2] + 1) < 0.1 for _, _, nor, _ in physics._segments)
    has_top_wall = any(abs(nor[2] - 1) < 0.1 for _, _, nor, _ in physics._segments)
    
    print(f"Has left wall (nor=(-1,0,0)):   {has_left_wall}")
    print(f"Has right wall (nor=(1,0,0)):    {has_right_wall}")
    print(f"Has bottom wall (nor=(0,0,-1)):  {has_bottom_wall}")
    print(f"Has top wall (nor=(0,0,1)):      {has_top_wall}")
    
    # Check simple collision case manually
    print(f"\n=== MANUAL COLLISION TEST ===")
    
    # Find the right wall segment
    right_wall_segments = [(i, seg) for i, seg in enumerate(physics._segments) 
                          if abs(seg[2][0] - 1) < 0.1]  # Normal ≈ (1,0,0)
    
    if right_wall_segments:
        seg_idx, (r_0, r_1, nor, tan) = right_wall_segments[0]
        print(f"Found right wall segment {seg_idx}:")
        print(f"  Start: ({r_0[0]:.3f}, {r_0[1]:.3f}, {r_0[2]:.3f})")
        print(f"  End:   ({r_1[0]:.3f}, {r_1[1]:.3f}, {r_1[2]:.3f})")
        print(f"  Normal: ({nor[0]:.3f}, {nor[1]:.3f}, {nor[2]:.3f})")
        
        # Test collision with ball moving right from center
        ball_pos = np.array([0.0, table.H + physics.ball_radius, 0.0])
        ball_vel = np.array([1.0, 0.0, 0.0])
        
        # Expected collision position
        expected_collision_x = table.W/2 - physics.ball_radius
        expected_time = expected_collision_x / ball_vel[0]
        
        print(f"  Ball starting at: ({ball_pos[0]:.3f}, {ball_pos[1]:.3f}, {ball_pos[2]:.3f})")
        print(f"  Ball velocity: ({ball_vel[0]:.3f}, {ball_vel[1]:.3f}, {ball_vel[2]:.3f})")
        print(f"  Expected collision X: {expected_collision_x:.3f}")
        print(f"  Expected collision time: {expected_time:.3f}")
        
        # Check if collision point is within segment
        collision_pos = ball_pos + expected_time * ball_vel
        # Project collision point onto segment
        seg_vec = r_1 - r_0
        seg_param = np.dot(collision_pos - r_0, tan) / np.dot(seg_vec, tan)
        
        print(f"  Collision position: ({collision_pos[0]:.3f}, {collision_pos[1]:.3f}, {collision_pos[2]:.3f})")
        print(f"  Segment parameter: {seg_param:.3f} (should be 0-1 for within segment)")
        
        within_segment = 0 <= seg_param <= 1
        print(f"  Within segment bounds: {within_segment}")
    else:
        print("No right wall segment found!")

if __name__ == "__main__":
    analyze_table_segments()