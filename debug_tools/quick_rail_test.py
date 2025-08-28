#!/usr/bin/env python3
"""
Quick test to validate X-direction rail collision fix
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pool_physics import PoolPhysics
from pool_physics.table import PoolTable
from pool_physics.events import BallSlidingEvent

def test_rail_collisions():
    """Quick test of both X and Z direction rail collisions"""
    
    print("QUICK RAIL COLLISION TEST")
    print("=" * 50)
    
    # Create physics engine with default parameters
    table = PoolTable()
    physics = PoolPhysics(table=table)
    
    # Test Z-direction (should work - this was already working)
    print("\n1. Testing Z-direction rail collision...")
    initial_pos = np.array([0.0, table.H, 0.0])  # Center of table
    initial_velocity = np.array([0.0, 0.0, 2.0])  # 2 m/s toward +Z rail
    
    physics.reset(balls_on_table=[0], ball_positions=np.array([initial_pos]))
    
    roll_event = BallSlidingEvent(
        t=0.0, i=0, r_0=initial_pos,
        v_0=initial_velocity, omega_0=np.array([0.0, 0.0, 0.0])
    )
    
    events = physics.add_event_sequence(roll_event)
    rail_collisions_z = [e for e in events if 'Segment' in type(e).__name__]
    
    print(f"   Initial velocity: {initial_velocity}")
    print(f"   Rail collisions detected: {len(rail_collisions_z)}")
    if rail_collisions_z:
        print(f"   First collision at t={rail_collisions_z[0].t:.3f}s")
        print(f"   ✅ Z-direction collision WORKS")
    else:
        print(f"   ❌ Z-direction collision FAILED")
    
    # Test X-direction (this should now work with our fix)
    print("\n2. Testing X-direction rail collision...")
    initial_pos = np.array([0.0, table.H, 0.0])  # Center of table
    initial_velocity = np.array([2.0, 0.0, 0.0])  # 2 m/s toward +X rail
    
    physics.reset(balls_on_table=[0], ball_positions=np.array([initial_pos]))
    
    roll_event = BallSlidingEvent(
        t=0.0, i=0, r_0=initial_pos,
        v_0=initial_velocity, omega_0=np.array([0.0, 0.0, 0.0])
    )
    
    events = physics.add_event_sequence(roll_event)
    rail_collisions_x = [e for e in events if 'Segment' in type(e).__name__]
    
    print(f"   Initial velocity: {initial_velocity}")
    print(f"   Rail collisions detected: {len(rail_collisions_x)}")
    if rail_collisions_x:
        print(f"   First collision at t={rail_collisions_x[0].t:.3f}s")
        print(f"   ✅ X-direction collision WORKS")
    else:
        print(f"   ❌ X-direction collision FAILED")
    
    # Test negative X-direction 
    print("\n3. Testing -X-direction rail collision...")
    initial_pos = np.array([0.0, table.H, 0.0])  # Center of table
    initial_velocity = np.array([-2.0, 0.0, 0.0])  # 2 m/s toward -X rail
    
    physics.reset(balls_on_table=[0], ball_positions=np.array([initial_pos]))
    
    roll_event = BallSlidingEvent(
        t=0.0, i=0, r_0=initial_pos,
        v_0=initial_velocity, omega_0=np.array([0.0, 0.0, 0.0])
    )
    
    events = physics.add_event_sequence(roll_event)
    rail_collisions_neg_x = [e for e in events if 'Segment' in type(e).__name__]
    
    print(f"   Initial velocity: {initial_velocity}")
    print(f"   Rail collisions detected: {len(rail_collisions_neg_x)}")
    if rail_collisions_neg_x:
        print(f"   First collision at t={rail_collisions_neg_x[0].t:.3f}s")
        print(f"   ✅ -X-direction collision WORKS")
    else:
        print(f"   ❌ -X-direction collision FAILED")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    z_works = len(rail_collisions_z) > 0
    x_works = len(rail_collisions_x) > 0
    neg_x_works = len(rail_collisions_neg_x) > 0
    
    print(f"Z-direction rails:  {'✅ WORKING' if z_works else '❌ BROKEN'}")
    print(f"X-direction rails:  {'✅ WORKING' if x_works else '❌ BROKEN'}")
    print(f"-X-direction rails: {'✅ WORKING' if neg_x_works else '❌ BROKEN'}")
    
    if x_works and neg_x_works:
        print("\n🎉 SUCCESS: X-direction rail collision fix is working!")
    else:
        print("\n💥 FAILURE: X-direction rail collisions still not working")
    
    # Assert results instead of returning them for pytest compatibility
    assert z_works, "Z-direction rail collision failed"
    assert x_works, "X-direction rail collision failed" 
    assert neg_x_works, "-X-direction rail collision failed"

if __name__ == "__main__":
    test_rail_collisions()
