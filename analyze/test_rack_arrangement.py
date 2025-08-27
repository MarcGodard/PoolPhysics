#!/usr/bin/env python3
"""
Test script to analyze the current rack arrangement
"""
import sys
import numpy as np
from pool_physics.table import PoolTable

def analyze_rack(rack_type='8-ball'):
    """Analyze the current rack arrangement"""
    table = PoolTable()
    positions = table.calc_racked_positions(rack_type=rack_type)
    
    print("Current Rack Arrangement:")
    print("=" * 40)
    
    # Print all ball positions
    for i in range(table.num_balls):
        x, y, z = positions[i]
        if i == 0:
            print(f"Ball {i:2d} (Cue): x={x:6.3f}, z={z:6.3f}")
        else:
            print(f"Ball {i:2d}:      x={x:6.3f}, z={z:6.3f}")
    
    print("\nRack Analysis:")
    print("-" * 20)
    
    # Find the apex ball (frontmost in rack)
    object_balls = positions[1:]  # Exclude cue ball
    apex_z = np.max(object_balls[:, 2])  # Most positive z (closest to head)
    apex_indices = np.where(np.abs(object_balls[:, 2] - apex_z) < 1e-6)[0] + 1
    print(f"Apex ball(s): {apex_indices} at z={apex_z:.3f}")
    
    # Find center ball (should be 8-ball in proper rack)
    # In a 5-row triangle: rows have 5,4,3,2,1 balls
    # The center ball is typically in the 3rd row, middle position
    center_z = positions[1:, 2]
    center_x = positions[1:, 0]
    
    # Find balls near center of rack
    z_center = np.mean([np.min(center_z), np.max(center_z)])
    x_center = 0.0  # Should be on centerline
    
    distances_from_center = np.sqrt((center_x)**2 + (center_z - z_center)**2)
    center_ball_idx = np.argmin(distances_from_center) + 1
    print(f"Current center ball: Ball {center_ball_idx}")
    
    # Check foot spot position
    foot_spot_z = -table.L / 4  # Standard foot spot
    print(f"Foot spot should be at z={foot_spot_z:.3f}")
    print(f"Apex ball is at z={apex_z:.3f}")
    print(f"Difference: {abs(apex_z - foot_spot_z):.3f}m")
    
    if rack_type == '8-ball':
        print(f"\nISSUES IDENTIFIED:")
        print(f"1. Apex ball should be at foot spot (z={foot_spot_z:.3f}), but is at z={apex_z:.3f}")
        print(f"2. Ball 8 should be in center position, but ball {center_ball_idx} is there")
        print(f"3. Current arrangement is purely geometric, not following 8-ball rules")
    elif rack_type == '9-ball':
        print(f"\n9-BALL RACK ANALYSIS:")
        print(f"- Ball 1 should be at apex (foot spot)")
        print(f"- Ball 9 should be in center of diamond")
        print(f"- Other balls can be in any position")
        print(f"- Should form a diamond shape (1-2-3-3 formation)")

if __name__ == "__main__":
    rack_type = sys.argv[1] if len(sys.argv) > 1 else '8-ball'
    analyze_rack(rack_type)
