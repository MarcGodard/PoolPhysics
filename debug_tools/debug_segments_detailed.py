#!/usr/bin/env python3
"""
Detailed segment analysis to debug X-direction rail collision issues
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pool_physics import PoolPhysics
from pool_physics.table import PoolTable

def analyze_segments_detailed():
    """Analyze segments in detail to understand collision detection"""
    
    table = PoolTable()
    physics = PoolPhysics(table=table)
    
    print("DETAILED SEGMENT ANALYSIS")
    print("=" * 60)
    print(f"Table dimensions: W={table.W:.3f}m, L={table.L:.3f}m, H={table.H:.3f}m")
    print(f"Ball radius: {physics.ball_radius:.3f}m")
    print(f"Expected rail collision height: {table.H + physics.ball_radius:.3f}m")
    
    print(f"\nTotal segments: {len(physics._segments)}")
    
    # Analyze each segment
    for i, (r_0, r_1, nor, tan) in enumerate(physics._segments):
        length = np.linalg.norm(r_1 - r_0)
        
        # Classify segment type by normal vector
        if abs(nor[0]) > 0.9:  # X-direction normal
            if nor[0] > 0:
                wall_type = "LEFT wall (nor=+X)"
            else:
                wall_type = "RIGHT wall (nor=-X)"
        elif abs(nor[2]) > 0.9:  # Z-direction normal
            if nor[2] > 0:
                wall_type = "BOTTOM wall (nor=+Z)"
            else:
                wall_type = "TOP wall (nor=-Z)"
        else:
            wall_type = "CORNER/POCKET"
            
        print(f"\nSegment {i:2d}: {wall_type}")
        print(f"  Start: ({r_0[0]:7.3f}, {r_0[1]:7.3f}, {r_0[2]:7.3f})")
        print(f"  End:   ({r_1[0]:7.3f}, {r_1[1]:7.3f}, {r_1[2]:7.3f})")
        print(f"  Normal:  ({nor[0]:6.3f}, {nor[1]:6.3f}, {nor[2]:6.3f})")
        print(f"  Length: {length:.3f}m")
        
        # Test collision for ball at center moving toward +X
        if abs(nor[0]) > 0.9 and nor[0] < 0:  # Right wall (normal points -X)
            ball_pos = np.array([0.0, table.H + physics.ball_radius, 0.0])
            ball_vel = np.array([1.0, 0.0, 0.0])
            
            # Calculate expected collision point
            # Ball center hits wall when ball_pos[0] + t*ball_vel[0] = wall_x - ball_radius
            wall_x = r_0[0]  # X-coordinate of wall
            collision_x = wall_x - physics.ball_radius
            expected_time = collision_x / ball_vel[0]
            collision_point = ball_pos + expected_time * ball_vel
            
            # Check if collision point is within segment bounds
            # Project collision point onto segment
            segment_vec = r_1 - r_0
            to_collision = collision_point - r_0
            segment_param = np.dot(to_collision, segment_vec) / np.dot(segment_vec, segment_vec)
            
            print(f"  ** COLLISION TEST for ball at center moving +X:")
            print(f"     Wall X-coordinate: {wall_x:.3f}")
            print(f"     Expected collision X: {collision_x:.3f}")
            print(f"     Expected time: {expected_time:.3f}s")
            print(f"     Collision point: ({collision_point[0]:.3f}, {collision_point[1]:.3f}, {collision_point[2]:.3f})")
            print(f"     Segment parameter: {segment_param:.3f} (should be 0-1)")
            print(f"     Within bounds: {0 <= segment_param <= 1}")
            
    # Test specific collision scenarios
    print("\n" + "=" * 60)
    print("COLLISION SCENARIO TESTS")
    print("=" * 60)
    
    # Test +X direction
    print("\n1. Testing +X direction collision detection:")
    ball_pos = np.array([0.0, table.H + physics.ball_radius, 0.0])
    ball_vel = np.array([2.0, 0.0, 0.0])
    
    print(f"   Ball position: ({ball_pos[0]:.3f}, {ball_pos[1]:.3f}, {ball_pos[2]:.3f})")
    print(f"   Ball velocity: ({ball_vel[0]:.3f}, {ball_vel[1]:.3f}, {ball_vel[2]:.3f})")
    
    # Find which segments should detect this collision
    right_wall_segments = []
    for i, (r_0, r_1, nor, tan) in enumerate(physics._segments):
        if abs(nor[0]) > 0.9 and nor[0] < 0:  # Right wall
            # Check if ball path intersects this segment
            wall_x = r_0[0]
            collision_x = wall_x - physics.ball_radius
            if collision_x > 0:  # Ball can reach this wall
                expected_time = collision_x / ball_vel[0]
                collision_point = ball_pos + expected_time * ball_vel
                
                # Check segment bounds
                segment_vec = r_1 - r_0
                to_collision = collision_point - r_0
                segment_param = np.dot(to_collision, segment_vec) / np.dot(segment_vec, segment_vec)
                
                if 0 <= segment_param <= 1:
                    right_wall_segments.append((i, expected_time, collision_point))
                    print(f"   Should hit segment {i} at t={expected_time:.3f}s")
    
    if not right_wall_segments:
        print("   ❌ No segments should detect +X collision - this is the problem!")
    else:
        print(f"   ✅ {len(right_wall_segments)} segments should detect +X collision")

if __name__ == "__main__":
    analyze_segments_detailed()
