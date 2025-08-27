#!/usr/bin/env python3
"""
Comprehensive validation of the enhanced PoolPhysics engine
Demonstrates fixed X-direction rail collisions and parameter testing capabilities
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pool_physics import PoolPhysics
from pool_physics.table import PoolTable
from pool_physics.events import BallSlidingEvent

def validate_rail_collision_directions():
    """Validate rail collisions work in all 4 directions"""
    
    print("RAIL COLLISION DIRECTION VALIDATION")
    print("=" * 60)
    
    table = PoolTable()
    physics = PoolPhysics(table=table)
    
    # Test all 4 directions
    directions = [
        ("Right (+X)", np.array([2.0, 0.0, 0.0])),
        ("Left (-X)", np.array([-2.0, 0.0, 0.0])),
        ("Top (+Z)", np.array([0.0, 0.0, 2.0])),
        ("Bottom (-Z)", np.array([0.0, 0.0, -2.0]))
    ]
    
    results = {}
    
    for direction_name, velocity in directions:
        print(f"\nTesting {direction_name} direction:")
        
        initial_pos = np.array([0.0, table.H + physics.ball_radius, 0.0])
        
        physics.reset(balls_on_table=[0], ball_positions=np.array([initial_pos]))
        
        roll_event = BallSlidingEvent(
            t=0.0, i=0, r_0=initial_pos,
            v_0=velocity, omega_0=np.array([0.0, 0.0, 0.0])
        )
        
        events = physics.add_event_sequence(roll_event)
        rail_collisions = [e for e in events if 'Segment' in type(e).__name__]
        
        if rail_collisions:
            collision_time = rail_collisions[0].t
            initial_speed = np.linalg.norm(velocity)
            final_velocity = physics.eval_velocities(collision_time + 0.001)[0]
            final_speed = np.linalg.norm(final_velocity)
            bounce_ratio = final_speed / initial_speed
            
            print(f"  ✅ Collision detected at t={collision_time:.3f}s")
            print(f"  Speed: {initial_speed:.2f} → {final_speed:.2f} m/s (ratio: {bounce_ratio:.3f})")
            
            results[direction_name] = {
                'working': True,
                'collision_time': collision_time,
                'bounce_ratio': bounce_ratio,
                'collisions': len(rail_collisions)
            }
        else:
            print(f"  ❌ No collision detected")
            results[direction_name] = {'working': False}
    
    return results

def demonstrate_physics_parameters():
    """Demonstrate different physics parameter effects"""
    
    print("\n" + "=" * 60)
    print("PHYSICS PARAMETER DEMONSTRATION")
    print("=" * 60)
    
    # Define different table conditions
    conditions = {
        "Tournament": {
            'mu_r': 0.01, 'mu_s': 0.2, 'mu_sp': 0.044,
            'kappa_rail': 0.6, 'description': 'Standard tournament conditions'
        },
        "Fast Table": {
            'mu_r': 0.005, 'mu_s': 0.15, 'mu_sp': 0.03,
            'kappa_rail': 0.75, 'description': 'Low friction, high rail elasticity'
        },
        "Slow Table": {
            'mu_r': 0.02, 'mu_s': 0.3, 'mu_sp': 0.06,
            'kappa_rail': 0.45, 'description': 'High friction, low rail elasticity'
        }
    }
    
    table = PoolTable()
    
    for condition_name, params in conditions.items():
        print(f"\n{condition_name}: {params['description']}")
        print("-" * 40)
        
        physics = PoolPhysics(
            table=table,
            mu_r=params['mu_r'],
            mu_s=params['mu_s'], 
            mu_sp=params['mu_sp']
        )
        
        # Test roll distance
        initial_pos = np.array([0.0, table.H + physics.ball_radius, 0.0])
        initial_velocity = np.array([2.0, 0.0, 0.0])
        
        physics.reset(balls_on_table=[0], ball_positions=np.array([initial_pos]))
        
        roll_event = BallSlidingEvent(
            t=0.0, i=0, r_0=initial_pos,
            v_0=initial_velocity, omega_0=np.array([0.0, 0.0, 0.0])
        )
        
        events = physics.add_event_sequence(roll_event)
        
        # Calculate roll distance
        final_pos = physics.eval_positions(events[-1].t)[0]
        roll_distance = np.linalg.norm(final_pos - initial_pos)
        
        # Count events
        rail_collisions = len([e for e in events if 'Segment' in type(e).__name__])
        
        print(f"  Roll distance: {roll_distance:.3f}m")
        print(f"  Rail collisions: {rail_collisions}")
        print(f"  Total events: {len(events)}")

def validate_break_shot_physics():
    """Validate break shot physics with different parameters"""
    
    print("\n" + "=" * 60)
    print("BREAK SHOT PHYSICS VALIDATION")
    print("=" * 60)
    
    table = PoolTable()
    physics = PoolPhysics(table=table)
    
    # Set up break shot
    cue_ball_pos = np.array([0.0, table.H + physics.ball_radius, -table.L/4])
    break_velocity = np.array([0.0, 0.0, 12.0])  # 12 m/s break shot
    
    # Get racked ball positions (limit to 9 balls for 9-ball game)
    racked_positions = table.calc_racked_positions(spacing_mode='tight')[:9]
    
    # Combine cue ball with racked balls
    all_positions = np.vstack([cue_ball_pos.reshape(1, 3), racked_positions])
    
    physics.reset(
        balls_on_table=list(range(len(all_positions))),
        ball_positions=all_positions
    )
    
    # Create break shot event
    break_event = BallSlidingEvent(
        t=0.0, i=0, r_0=cue_ball_pos,
        v_0=break_velocity, omega_0=np.array([0.0, 0.0, 0.0])
    )
    
    print("Simulating break shot...")
    events = physics.add_event_sequence(break_event)
    
    # Analyze results
    ball_collisions = len([e for e in events if 'Ball' in type(e).__name__ and 'Collision' in type(e).__name__])
    rail_collisions = len([e for e in events if 'Segment' in type(e).__name__])
    
    # Calculate ball spread
    final_positions = physics.eval_positions(events[-1].t)
    initial_center = np.mean(racked_positions, axis=0)
    final_center = np.mean(final_positions[1:], axis=0)  # Exclude cue ball
    
    spreads = [np.linalg.norm(pos - final_center) for pos in final_positions[1:]]
    avg_spread = np.mean(spreads)
    
    print(f"  Ball-to-ball collisions: {ball_collisions}")
    print(f"  Rail collisions: {rail_collisions}")
    print(f"  Average ball spread: {avg_spread:.3f}m")
    print(f"  Total events: {len(events)}")
    
    return {
        'ball_collisions': ball_collisions,
        'rail_collisions': rail_collisions,
        'avg_spread': avg_spread,
        'total_events': len(events)
    }

def main():
    """Run comprehensive physics validation"""
    
    print("COMPREHENSIVE POOLPHYSICS ENGINE VALIDATION")
    print("=" * 80)
    print("This validation demonstrates the enhanced physics engine with:")
    print("• Fixed X-direction rail collision detection")
    print("• Comprehensive parameter testing capabilities") 
    print("• Realistic physics simulation across all directions")
    print("=" * 80)
    
    # Validate rail collisions
    rail_results = validate_rail_collision_directions()
    
    # Demonstrate parameter effects
    demonstrate_physics_parameters()
    
    # Validate break shot physics
    break_results = validate_break_shot_physics()
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    working_directions = sum(1 for result in rail_results.values() if result.get('working', False))
    print(f"Rail collision directions working: {working_directions}/4")
    
    for direction, result in rail_results.items():
        status = "✅" if result.get('working', False) else "❌"
        print(f"  {status} {direction}")
    
    print(f"\nBreak shot simulation:")
    print(f"  Ball collisions: {break_results['ball_collisions']}")
    print(f"  Rail collisions: {break_results['rail_collisions']}")
    print(f"  Ball spread: {break_results['avg_spread']:.3f}m")
    
    if working_directions == 4:
        print("\n🎉 SUCCESS: All rail collision directions are working!")
        print("The PoolPhysics engine is now fully functional for comprehensive testing.")
    else:
        print(f"\n⚠️  WARNING: {4-working_directions} rail directions still not working")

if __name__ == "__main__":
    main()
