#!/usr/bin/env python3
"""
Multi-run break analysis script to study the impact of random ball spacing.

Runs multiple break simulations and outputs statistical analysis of the results.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from pool_physics import PoolPhysics
from pool_physics.table import PoolTable
from pool_physics.events import BallCollisionEvent, RailCollisionEvent, BallSlidingEvent
from typing import List


def _get_balls_for_rack_type(rack_type: str) -> List[int]:
    """Get list of ball IDs for specified rack type."""
    if rack_type == '9-ball':
        return list(range(10))  # Balls 0-9
    elif rack_type == '10-ball':
        return list(range(11))  # Balls 0-10
    else:  # 8-ball
        return list(range(16))  # Balls 0-15


def run_break_simulation(rack_type='8-ball', break_speed=12.0, max_time=10.0):
    """
    Run a single break simulation and return statistics.
    
    Args:
        rack_type: Type of rack ('8-ball', '9-ball', '10-ball')
        break_speed: Break shot speed in m/s
        max_time: Maximum simulation time in seconds
        
    Returns:
        dict: Statistics from the break simulation
    """
    # Initialize physics and table
    physics = PoolPhysics()
    table = PoolTable()
    
    # Get rack positions for specified rack type
    initial_positions = table.calc_racked_positions(rack_type=rack_type)
    
    # Determine which balls are on table based on rack type
    balls_on_table = _get_balls_for_rack_type(rack_type)
    
    # Reset physics with appropriate balls
    physics.reset(balls_on_table=balls_on_table, ball_positions=initial_positions)
    
    # Set up break shot parameters
    cue_ball_pos = initial_positions[0].copy()  # Cue ball at head spot
    target_pos = initial_positions[1].copy()    # Target apex ball at foot spot
    
    # Calculate direction vector from cue ball to target
    direction = target_pos - cue_ball_pos
    direction[1] = 0.0  # Keep shot horizontal (y=0)
    direction_norm = np.linalg.norm(direction)
    
    if direction_norm == 0:
        raise ValueError("Invalid table setup: cue ball and target ball at same position")
        
    direction_unit = direction / direction_norm
    
    # Set up cue ball velocity (aimed at apex ball)
    v_0 = direction_unit * break_speed
    
    # Add slight backspin for realistic break shot
    omega_0 = np.array([0.0, -5.0, 0.0])
    
    # Create and execute break event
    break_event = BallSlidingEvent(
        t=0.0,
        i=0,
        r_0=cue_ball_pos,
        v_0=v_0,
        omega_0=omega_0
    )
    
    # Run simulation
    events = physics.add_event_sequence(break_event)
    
    # Get final positions after simulation settles
    final_positions = physics.eval_positions(max_time)
    
    # Count events
    ball_collisions = len([e for e in events if isinstance(e, BallCollisionEvent)])
    rail_collisions = len([e for e in events if isinstance(e, RailCollisionEvent)])
    
    # Also check for SegmentCollisionEvent (actual rail collisions)
    segment_collisions = len([e for e in events if type(e).__name__ == 'SegmentCollisionEvent'])
    if segment_collisions > 0 and rail_collisions == 0:
        rail_collisions = segment_collisions
    
    total_events = len(events)
    
    # Count balls remaining on table by checking positions
    balls_on_table_final = 0
    for ball_id in balls_on_table:
        if ball_id < len(final_positions):
            x, y, z = final_positions[ball_id]
            
            # Check if ball is still on table (not pocketed or off table)
            below_table = y < table.H - table.ball_radius
            outside_x = abs(x) > table.W/2 + table.ball_radius
            outside_z = abs(z) > table.L/2 + table.ball_radius
            
            if not (below_table or outside_x or outside_z):
                balls_on_table_final += 1
    
    # Calculate pocketed balls
    total_balls = len(balls_on_table)
    balls_pocketed = total_balls - balls_on_table_final
    balls_remaining = balls_on_table_final
    
    return {
        'total_events': total_events,
        'ball_collisions': ball_collisions,
        'rail_collisions': rail_collisions,
        'balls_pocketed': balls_pocketed,
        'balls_remaining': balls_remaining
    }


def analyze_break_variations(rack_type='8-ball', break_speed=12.0, num_runs=10):
    """
    Run multiple break simulations and analyze the variations.
    
    Args:
        rack_type: Type of rack to analyze
        break_speed: Break shot speed in m/s
        num_runs: Number of simulations to run
    """
    print(f"🎱 Break Variation Analysis - {rack_type.title()}")
    print("=" * 60)
    print(f"Running {num_runs} simulations with random spacing (0.01-0.2mm)...")
    print()
    
    results = []
    
    # Run multiple simulations
    for i in range(num_runs):
        print(f"Run {i+1:2d}...", end=" ", flush=True)
        try:
            result = run_break_simulation(rack_type, break_speed)
            results.append(result)
            print("✅")
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    if not results:
        print("No successful simulations completed.")
        return
    
    print(f"\n📊 Results Summary ({len(results)} successful runs):")
    print()
    
    # Print table header
    print("| Run | Total Events | Ball Collisions | Rail Collisions | Pocketed | Remaining |")
    print("|-----|-------------|----------------|-----------------|----------|-----------|")
    
    # Print individual results
    for i, result in enumerate(results, 1):
        print(f"| {i:3d} | {result['total_events']:11d} | "
              f"{result['ball_collisions']:14d} | {result['rail_collisions']:14d} | "
              f"{result['balls_pocketed']:8d} | {result['balls_remaining']:9d} |")
    
    # Calculate statistics
    total_events = [r['total_events'] for r in results]
    ball_collisions = [r['ball_collisions'] for r in results]
    rail_collisions = [r['rail_collisions'] for r in results]
    balls_pocketed = [r['balls_pocketed'] for r in results]
    balls_remaining = [r['balls_remaining'] for r in results]
    
    print("|-----|-------------|----------------|-----------------|----------|-----------|")
    print(f"| Min | {min(total_events):11d} | {min(ball_collisions):14d} | "
          f"{min(rail_collisions):14d} | {min(balls_pocketed):8d} | {min(balls_remaining):9d} |")
    print(f"| Max | {max(total_events):11d} | {max(ball_collisions):14d} | "
          f"{max(rail_collisions):14d} | {max(balls_pocketed):8d} | {max(balls_remaining):9d} |")
    print(f"| Avg | {np.mean(total_events):11.1f} | {np.mean(ball_collisions):14.1f} | "
          f"{np.mean(rail_collisions):14.1f} | {np.mean(balls_pocketed):8.1f} | {np.mean(balls_remaining):9.1f} |")
    print(f"| Std | {np.std(total_events):11.1f} | {np.std(ball_collisions):14.1f} | "
          f"{np.std(rail_collisions):14.1f} | {np.std(balls_pocketed):8.1f} | {np.std(balls_remaining):9.1f} |")
    
    print()
    print("📈 Variation Analysis:")
    print(f"   • Total Events: {min(total_events)}-{max(total_events)} "
          f"(range: {max(total_events) - min(total_events)})")
    print(f"   • Ball Collisions: {min(ball_collisions)}-{max(ball_collisions)} "
          f"(range: {max(ball_collisions) - min(ball_collisions)})")
    print(f"   • Rail Collisions: {min(rail_collisions)}-{max(rail_collisions)} "
          f"(range: {max(rail_collisions) - min(rail_collisions)})")
    print(f"   • Balls Pocketed: {min(balls_pocketed)}-{max(balls_pocketed)} "
          f"(range: {max(balls_pocketed) - min(balls_pocketed)})")
    
    # Coefficient of variation (std/mean) as percentage
    cv_events = (np.std(total_events) / np.mean(total_events)) * 100
    cv_pockets = (np.std(balls_pocketed) / max(np.mean(balls_pocketed), 0.1)) * 100
    
    print()
    print("🎯 Consistency Metrics:")
    print(f"   • Event Variability: {cv_events:.1f}% (lower = more consistent)")
    print(f"   • Pocket Variability: {cv_pockets:.1f}% (lower = more predictable)")


def main():
    """Main function with command-line interface."""
    if len(sys.argv) > 1:
        rack_type = sys.argv[1]
        break_speed = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
        num_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    else:
        rack_type = '8-ball'
        break_speed = 12.0
        num_runs = 10
    
    analyze_break_variations(rack_type, break_speed, num_runs)


if __name__ == "__main__":
    main()
