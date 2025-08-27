#!/usr/bin/env python3
"""
Debug script to examine all events in detail
"""
import numpy as np
from pool_physics.table import PoolTable
from pool_physics import PoolPhysics
from pool_physics.events import BallSlidingEvent

def debug_all_events():
    """Debug all events to see what types are generated"""
    
    # Initialize physics
    physics = PoolPhysics()
    table = PoolTable()
    
    # Get 8-ball rack positions
    initial_positions = table.calc_racked_positions(rack_type='8-ball')
    balls_on_table = list(range(16))  # 0-15
    
    # Reset physics with all balls
    physics.reset(balls_on_table=balls_on_table, ball_positions=initial_positions)
    
    # Set up break shot
    cue_ball_pos = initial_positions[0].copy()
    target_pos = initial_positions[1].copy()
    
    direction = target_pos - cue_ball_pos
    direction[1] = 0  # Keep y=0 for horizontal shot
    direction_norm = np.linalg.norm(direction)
    direction_unit = direction / direction_norm
    
    # Create break event at 10 m/s
    v_0 = direction_unit * 10.0
    omega_0 = np.array([0.0, -5.0, 0.0])
    
    break_event = BallSlidingEvent(
        t=0.0,
        i=0,
        r_0=cue_ball_pos,
        v_0=v_0,
        omega_0=omega_0
    )
    
    # Run simulation
    events = physics.add_event_sequence(break_event)
    
    # Examine all event types
    event_types = {}
    for event in events:
        event_type = type(event).__name__
        if event_type not in event_types:
            event_types[event_type] = []
        event_types[event_type].append(event)
    
    print(f"Total events: {len(events)}")
    print(f"\nEvent types found:")
    
    for event_type, event_list in event_types.items():
        print(f"  {event_type}: {len(event_list)} events")
        
        # Show first few events of each type
        for i, event in enumerate(event_list[:3]):
            print(f"    Event {i}: t={event.t:.4f}, ", end="")
            if hasattr(event, 'i'):
                print(f"ball={event.i}, ", end="")
            if hasattr(event, 'side'):
                print(f"side={event.side}, ", end="")
            if hasattr(event, 'e_i') and hasattr(event.e_i, 'i'):
                print(f"ball={event.e_i.i}, ", end="")
            print()
    
    # Check for specific rail/rail collision events
    from pool_physics.events import RailCollisionEvent
    print(f"\nSpecific RailCollisionEvent check:")
    rail_events = [e for e in events if isinstance(e, RailCollisionEvent)]
    print(f"RailCollisionEvent instances: {len(rail_events)}")
    
    # Check all possible names for rail-related events
    rail_like_events = [e for e in events if 'rail' in type(e).__name__.lower() or 'wall' in type(e).__name__.lower()]
    print(f"Rail/Wall-like events: {len(rail_like_events)}")
    
    return events, event_types

if __name__ == "__main__":
    debug_all_events()