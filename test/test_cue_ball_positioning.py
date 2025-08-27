#!/usr/bin/env python3
"""
Test script to verify random cue ball positioning functionality
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pool_physics.table import PoolTable

def test_cue_ball_positioning():
    """Test cue ball positioning with different spacing modes"""
    table = PoolTable()
    
    print("Testing Cue Ball Positioning")
    print("=" * 40)
    
    # Test fixed positioning (should be at head spot)
    positions_fixed = table.calc_racked_positions(
        rack_type='9-ball', 
        spacing_mode='fixed'
    )
    cue_ball_fixed = positions_fixed[0]
    print(f"Fixed mode - Cue ball position: x={cue_ball_fixed[0]:.6f}m, z={cue_ball_fixed[2]:.6f}m")
    
    # Test random positioning with seed (should be reproducible)
    positions_random1 = table.calc_racked_positions(
        rack_type='9-ball', 
        spacing_mode='random', 
        seed=42
    )
    cue_ball_random1 = positions_random1[0]
    print(f"Random mode (seed=42) - Cue ball position: x={cue_ball_random1[0]:.6f}m, z={cue_ball_random1[2]:.6f}m")
    
    # Test same seed again (should be identical)
    positions_random2 = table.calc_racked_positions(
        rack_type='9-ball', 
        spacing_mode='random', 
        seed=42
    )
    cue_ball_random2 = positions_random2[0]
    print(f"Random mode (seed=42 again) - Cue ball position: x={cue_ball_random2[0]:.6f}m, z={cue_ball_random2[2]:.6f}m")
    
    # Test different seed (should be different)
    positions_random3 = table.calc_racked_positions(
        rack_type='9-ball', 
        spacing_mode='random', 
        seed=123
    )
    cue_ball_random3 = positions_random3[0]
    print(f"Random mode (seed=123) - Cue ball position: x={cue_ball_random3[0]:.6f}m, z={cue_ball_random3[2]:.6f}m")
    
    print("\nValidation:")
    print("-" * 20)
    
    # Verify reproducibility
    is_reproducible = np.allclose(cue_ball_random1, cue_ball_random2)
    print(f"✅ Reproducible with same seed: {is_reproducible}")
    
    # Verify different seeds produce different results
    is_different = not np.allclose(cue_ball_random1, cue_ball_random3)
    print(f"✅ Different seeds produce different positions: {is_different}")
    
    # Verify offsets are within 2.5mm limit
    head_spot_z = 0.25 * table.L
    max_offset = 2.5e-3  # 2.5mm
    
    # Calculate actual offsets from head spot
    x_offset1 = cue_ball_random1[0]  # Can be positive or negative
    z_offset1 = cue_ball_random1[2] - head_spot_z  # Should be negative (toward head rail)
    
    x_within_limit = abs(x_offset1) <= max_offset
    z_within_limit = -max_offset <= z_offset1 <= 0  # Should be between -2.5mm and 0
    behind_head_string = cue_ball_random1[2] >= head_spot_z - max_offset
    
    print(f"✅ X offset within 2.5mm: {x_within_limit} (offset: {x_offset1*1000:+.2f}mm)")
    print(f"✅ Z offset within 2.5mm: {z_within_limit} (offset: {z_offset1*1000:+.2f}mm)")
    print(f"✅ Cue ball behind head string: {behind_head_string}")
    
    # Test without seed (should be different each time)
    print(f"\nTesting multiple runs without seed:")
    for i in range(3):
        positions_no_seed = table.calc_racked_positions(
            rack_type='9-ball', 
            spacing_mode='random'
        )
        cue_ball_no_seed = positions_no_seed[0]
        print(f"Run {i+1} - Cue ball position: x={cue_ball_no_seed[0]:.6f}m, z={cue_ball_no_seed[2]:.6f}m")

if __name__ == "__main__":
    test_cue_ball_positioning()
