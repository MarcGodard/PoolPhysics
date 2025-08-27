#!/usr/bin/env python3
"""
Demonstrate basic physics data captured in PoolPhysics system:
- Initial spin values (omega_0)
- Initial velocities 
- Event counting and collision details
"""
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pool_physics.table import PoolTable
from pool_physics import PoolPhysics
from pool_physics.events import BallSlidingEvent, BallCollisionEvent, RailCollisionEvent

def demonstrate_physics_data():
    """Show comprehensive physics data capture"""
    
    print("=" * 60)
    print("POOL PHYSICS - BASIC DATA DEMONSTRATION")
    print("=" * 60)
    
    # Initialize physics
    physics = PoolPhysics()
    table = PoolTable()
    
    # Get 8-ball rack positions
    initial_positions = table.calc_racked_positions(rack_type='8-ball')
    balls_on_table = list(range(16))  # 0-15
    
    # Reset physics with all balls
    physics.reset(balls_on_table=balls_on_table, ball_positions=initial_positions)
    
    # Set up break shot parameters
    cue_ball_pos = initial_positions[0].copy()
    target_pos = initial_positions[1].copy()
    
    # Calculate direction vector
    direction = target_pos - cue_ball_pos
    direction[1] = 0  # Keep y=0 for horizontal shot
    direction_norm = np.linalg.norm(direction)
    direction_unit = direction / direction_norm
    
    # DEMONSTRATION 1: INITIAL SPIN VALUES (omega_0)
    print("\n1. INITIAL SPIN VALUES (Angular Velocity)")
    print("-" * 40)
    
    # Different spin examples
    spin_examples = {
        "No Spin": np.array([0.0, 0.0, 0.0]),
        "Backspin": np.array([0.0, -5.0, 0.0]),
        "Topspin": np.array([0.0, 5.0, 0.0]),
        "Left English": np.array([-3.0, 0.0, 0.0]),
        "Right English": np.array([3.0, 0.0, 0.0]),
        "Combined (Back+Right)": np.array([2.0, -4.0, 0.0])
    }
    
    for spin_name, omega_0 in spin_examples.items():
        print(f"{spin_name:20}: omega_0 = [{omega_0[0]:5.1f}, {omega_0[1]:5.1f}, {omega_0[2]:5.1f}] rad/s")
        print(f"{'':20}  |omega| = {np.linalg.norm(omega_0):5.2f} rad/s")
    
    # DEMONSTRATION 2: INITIAL VELOCITIES
    print("\n2. INITIAL VELOCITY CALCULATIONS")
    print("-" * 40)
    
    break_speeds = [8.0, 12.0, 16.0, 20.0]
    
    for speed in break_speeds:
        v_0 = direction_unit * speed
        v_magnitude = np.linalg.norm(v_0)
        print(f"Break speed {speed:4.1f} m/s: v_0 = [{v_0[0]:6.3f}, {v_0[1]:6.3f}, {v_0[2]:6.3f}] m/s")
        print(f"{'':20}  |v_0| = {v_magnitude:5.2f} m/s")
        print(f"{'':20}  Direction: [{direction_unit[0]:6.3f}, {direction_unit[1]:6.3f}, {direction_unit[2]:6.3f}]")
    
    # DEMONSTRATION 3: RUN SIMULATION WITH DETAILED TRACKING
    print("\n3. SIMULATION WITH EVENT TRACKING")
    print("-" * 40)
    
    # Use moderate break speed with backspin
    break_speed = 12.0
    v_0 = direction_unit * break_speed
    omega_0 = np.array([0.0, -5.0, 0.0])  # Backspin
    
    print(f"Break shot setup:")
    print(f"  Cue ball position: [{cue_ball_pos[0]:6.3f}, {cue_ball_pos[1]:6.3f}, {cue_ball_pos[2]:6.3f}] m")
    print(f"  Target position:   [{target_pos[0]:6.3f}, {target_pos[1]:6.3f}, {target_pos[2]:6.3f}] m")
    print(f"  Initial velocity:  [{v_0[0]:6.3f}, {v_0[1]:6.3f}, {v_0[2]:6.3f}] m/s")
    print(f"  Initial spin:      [{omega_0[0]:6.3f}, {omega_0[1]:6.3f}, {omega_0[2]:6.3f}] rad/s")
    
    # Create and run break event
    break_event = BallSlidingEvent(
        t=0.0,
        i=0,
        r_0=cue_ball_pos,
        v_0=v_0,
        omega_0=omega_0
    )
    
    print(f"\nRunning simulation...")
    events = physics.add_event_sequence(break_event)
    
    # DEMONSTRATION 4: EVENT COUNTING AND ANALYSIS
    print("\n4. EVENT COUNTING AND COLLISION ANALYSIS")
    print("-" * 40)
    
    # Count event types
    event_types = {}
    collision_events = []
    rail_events = []
    
    for event in events:
        event_type = type(event).__name__
        if event_type not in event_types:
            event_types[event_type] = []
        event_types[event_type].append(event)
        
        # Collect collision events for detailed analysis
        if 'Collision' in event_type:
            collision_events.append(event)
        if 'Segment' in event_type or 'Rail' in event_type:
            rail_events.append(event)
    
    print(f"Total events generated: {len(events)}")
    print(f"\nEvent type breakdown:")
    
    for event_type, event_list in sorted(event_types.items()):
        print(f"  {event_type:25}: {len(event_list):3d} events")
    
    # Detailed collision analysis
    print(f"\nCollision Event Details:")
    print(f"  Ball-to-ball collisions: {len([e for e in events if 'SimpleBallCollision' in type(e).__name__])}")
    print(f"  Rail/wall collisions:    {len([e for e in events if 'Segment' in type(e).__name__])}")
    
    # Show first few collision events with timing
    print(f"\nFirst 5 collision events:")
    collision_count = 0
    for event in events:
        if 'Collision' in type(event).__name__ or 'Segment' in type(event).__name__:
            collision_count += 1
            if collision_count <= 5:
                event_type = type(event).__name__
                print(f"  {collision_count}. t={event.t:6.4f}s - {event_type}", end="")
                if hasattr(event, 'i'):
                    print(f" (ball {event.i})", end="")
                if hasattr(event, 'j') and event.j is not None:
                    print(f" with ball {event.j}", end="")
                print()
    
    # DEMONSTRATION 5: PHYSICS STATE AT DIFFERENT TIMES
    print("\n5. PHYSICS STATE EVOLUTION")
    print("-" * 40)
    
    time_points = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
    
    print(f"Ball positions and velocities over time:")
    print(f"{'Time':>6} {'Ball':>4} {'Position (x,y,z)':>25} {'Velocity (x,y,z)':>25}")
    print("-" * 70)
    
    for t in time_points:
        positions = physics.eval_positions(t)
        velocities = physics.eval_velocities(t)
        
        # Show cue ball (ball 0) state
        if len(positions) > 0:
            pos = positions[0]
            vel = velocities[0] if len(velocities) > 0 else np.array([0,0,0])
            print(f"{t:6.1f} {0:4d} [{pos[0]:6.3f},{pos[1]:6.3f},{pos[2]:6.3f}] [{vel[0]:6.3f},{vel[1]:6.3f},{vel[2]:6.3f}]")
    
    # DEMONSTRATION 6: ENERGY AND MOMENTUM TRACKING
    print("\n6. ENERGY AND MOMENTUM ANALYSIS")
    print("-" * 40)
    
    # Calculate initial kinetic energy
    initial_ke_trans = 0.5 * physics.ball_mass * np.linalg.norm(v_0)**2
    initial_ke_rot = 0.5 * physics.ball_I * np.linalg.norm(omega_0)**2
    total_initial_ke = initial_ke_trans + initial_ke_rot
    
    print(f"Initial energy analysis:")
    print(f"  Translational KE: {initial_ke_trans:6.3f} J")
    print(f"  Rotational KE:    {initial_ke_rot:6.3f} J")
    print(f"  Total initial KE: {total_initial_ke:6.3f} J")
    
    # Show momentum
    initial_momentum = physics.ball_mass * v_0
    initial_angular_momentum = physics.ball_I * omega_0
    
    print(f"\nInitial momentum:")
    print(f"  Linear momentum:  [{initial_momentum[0]:6.3f}, {initial_momentum[1]:6.3f}, {initial_momentum[2]:6.3f}] kg⋅m/s")
    print(f"  Angular momentum: [{initial_angular_momentum[0]:6.3f}, {initial_angular_momentum[1]:6.3f}, {initial_angular_momentum[2]:6.3f}] kg⋅m²/s")
    
    print("\n" + "=" * 60)
    print("SUMMARY: The physics engine captures rich data including:")
    print("• Initial conditions (position, velocity, spin)")
    print("• Event sequences (collisions, transitions, rest)")
    print("• Time evolution of all ball states")
    print("• Energy and momentum conservation")
    print("• Detailed collision timing and participants")
    print("=" * 60)
    
    return events, physics

if __name__ == "__main__":
    demonstrate_physics_data()
