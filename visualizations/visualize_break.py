#!/usr/bin/env python3
"""
Visualize pool break shots for different rack types
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pool_physics.table import PoolTable
from pool_physics import PoolPhysics
from pool_physics.events import BallSlidingEvent
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.table_renderer import draw_pool_table, draw_balls, draw_pocketed_balls

def simulate_break(rack_type='8-ball', break_speed=12.0, spacing_mode='random', seed=None):
    """
    Simulate a break shot for the given rack type
    
    Args:
        rack_type: '8-ball', '9-ball', or '10-ball'
        break_speed: cue ball speed in m/s
        spacing_mode: Ball spacing mode ('random', 'fixed', or 'tight')
        seed: Random seed for reproducible spacing (only used with 'random' mode)
    """
    print(f"\nSimulating {rack_type} break at {break_speed} m/s...")
    
    # Initialize physics
    physics = PoolPhysics()
    table = PoolTable()
    
    # Get rack positions
    initial_positions = table.calc_racked_positions(rack_type=rack_type, spacing_mode=spacing_mode, seed=seed)
    
    # Determine which balls are on table based on rack type
    if rack_type == '9-ball':
        balls_on_table = list(range(10))  # 0-9
    elif rack_type == '10-ball':
        balls_on_table = list(range(11))  # 0-10
    else:  # 8-ball
        balls_on_table = list(range(16))  # 0-15
    
    # Reset physics with appropriate balls
    physics.reset(balls_on_table=balls_on_table, ball_positions=initial_positions)
    
    # Set up break shot parameters
    cue_ball_pos = initial_positions[0].copy()  # Head spot position
    
    # Calculate target position (apex ball)
    target_pos = initial_positions[1].copy()  # Ball 1 at foot spot
    
    # Calculate direction vector from cue ball to target
    direction = target_pos - cue_ball_pos
    direction[1] = 0  # Keep y=0 for horizontal shot
    direction_norm = np.linalg.norm(direction)
    direction_unit = direction / direction_norm
    
    # Set up cue ball velocity (aimed at apex ball)
    v_0 = direction_unit * break_speed
    
    # Add slight downward english (hit below center)
    # This creates forward roll and helps with power transfer
    omega_0 = np.array([0.0, -5.0, 0.0])  # Slight backspin initially, then forward roll
    
    # Create break event
    break_event = BallSlidingEvent(
        t=0.0,
        i=0,
        r_0=cue_ball_pos,
        v_0=v_0,
        omega_0=omega_0
    )
    
    print(f"Break direction: {direction_unit}")
    print(f"Break velocity: {v_0}")
    print(f"Initial spin: {omega_0}")
    
    # Run simulation
    print("Running break simulation...")
    events = physics.add_event_sequence(break_event)
    
    print(f"Generated {len(events)} events")
    
    # Get final positions after simulation settles
    final_time = 10.0  # Let simulation run for 10 seconds
    final_positions = physics.eval_positions(final_time)
    
    return physics, initial_positions, final_positions, events, balls_on_table

def visualize_break_comparison(rack_type='8-ball', break_speed=12.0, spacing_mode='random', seed=None):
    """Create before/after visualization of break shot"""
    
    # Run simulation
    physics, initial_pos, final_pos, events, balls_on_table = simulate_break(rack_type, break_speed, spacing_mode, seed)
    table = PoolTable()
    
    # Create figure with before/after subplots
    plt.close('all')  # Close any existing figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.set_dpi(100)
    
    # Table dimensions
    table_width = table.W
    table_length = table.L
    ball_radius = table.ball_radius
    
    # Colors for balls
    colors = plt.cm.Set3(np.linspace(0, 1, 16))
    
    for ax, positions, title in [(ax1, initial_pos, 'Before Break'), 
                                 (ax2, final_pos, 'After Break')]:
        
        # Draw table using unified renderer (standard style for break comparison)
        style_info = draw_pool_table(ax, table, style='standard', show_pockets=True, show_spots=True, show_grid=True, show_pocketed_area=True)
        
        # Draw balls using unified renderer with ball filtering for physics simulation
        filtered_positions = []
        filtered_balls = []
        for ball_id in balls_on_table:
            if ball_id < len(positions):
                x, y, z = positions[ball_id]
                
                # Skip balls that are off table (pocketed or escaped through boundaries)
                below_table = y < table.H - ball_radius
                outside_x = abs(x) > table.W/2 + ball_radius
                outside_z = abs(z) > table.L/2 + ball_radius
                
                if below_table or outside_x or outside_z:
                    continue
                
                # Skip balls at origin for 9-ball and 10-ball (unused balls)
                if rack_type in ['9-ball', '10-ball'] and ball_id > 0 and abs(x) < 1e-6 and abs(z) < 1e-6:
                    continue
                
                filtered_positions.append([x, y, z])
                filtered_balls.append(ball_id)
        
        # Draw the filtered balls on table
        if filtered_positions:
            filtered_positions = np.array(filtered_positions)
            for i, ball_id in enumerate(filtered_balls):
                x, _, z = filtered_positions[i]
                circle = plt.Circle((x, z), ball_radius, 
                                  color=colors[ball_id], alpha=style_info['ball_alpha'],
                                  edgecolor=style_info['ball_edge_color'], 
                                  linewidth=style_info['ball_edge_width'])
                ax.add_patch(circle)
                
                # Add ball number
                ax.text(x, z, str(ball_id), ha='center', va='center', 
                       fontsize=style_info['font_size'], fontweight='bold')
        
        # Draw pocketed balls below the table
        draw_pocketed_balls(ax, positions, balls_on_table, ball_radius, colors, style_info, table, rack_type)
        
        # Set axis labels and title
        ax.set_xlabel('X Position (meters)')
        ax.set_ylabel('Z Position (meters)')
        ax.set_title(f'{title}\n{rack_type.title()} at {break_speed} m/s')
    
    plt.tight_layout()
    
    # Save visualization
    import os
    os.makedirs('images', exist_ok=True)
    filename = f'images/{rack_type}_break_comparison.png'
    plt.savefig(filename, dpi=100, bbox_inches=None, facecolor='white')
    print(f"Break comparison saved as '{filename}'")
    
    # Print break statistics
    print(f"\nBreak Statistics:")
    print(f"Rack type: {rack_type}")
    print(f"Break speed: {break_speed} m/s")
    print(f"Total events: {len(events)}")
    
    # Count ball types in events
    from pool_physics.events import BallCollisionEvent, RailCollisionEvent
    ball_collisions = len([e for e in events if isinstance(e, BallCollisionEvent)])
    rail_collisions = len([e for e in events if isinstance(e, RailCollisionEvent)])
    
    # Also check for SegmentCollisionEvent (actual rail collisions)
    segment_collisions = len([e for e in events if type(e).__name__ == 'SegmentCollisionEvent'])
    if segment_collisions > 0 and rail_collisions == 0:
        rail_collisions = segment_collisions
    
    print(f"Ball-to-ball collisions: {ball_collisions}")
    print(f"Rail collisions: {rail_collisions}")
    
    # Count balls remaining on table
    balls_on_table_final = 0
    for ball_id in balls_on_table:
        if ball_id < len(final_pos):
            x, y, z = final_pos[ball_id]
            
            # Skip balls at origin for 9-ball and 10-ball (unused balls)
            if rack_type in ['9-ball', '10-ball'] and ball_id > 0 and abs(x) < 1e-6 and abs(z) < 1e-6:
                continue
                
            # Check if ball is still on table
            below_table = y < table.H - ball_radius
            outside_x = abs(x) > table.W/2 + ball_radius
            outside_z = abs(z) > table.L/2 + ball_radius
            
            if not (below_table or outside_x or outside_z):
                balls_on_table_final += 1
    
    # Calculate initial count properly (excluding unused balls)
    balls_initially_active = len(balls_on_table)
    if rack_type == '9-ball':
        balls_initially_active = 10  # 0-9 balls
    elif rack_type == '10-ball':
        balls_initially_active = 11  # 0-10 balls
    
    balls_pocketed = balls_initially_active - balls_on_table_final
    print(f"Balls pocketed: {balls_pocketed}")
    print(f"Balls remaining: {balls_on_table_final}")
    
    plt.show()
    
    return physics, events

def visualize_all_breaks(break_speed=12.0, spacing_mode='random', seed=None):
    """Visualize break shots for all rack types"""
    for rack_type in ['8-ball', '9-ball', '10-ball']:
        print(f"\n{'='*60}")
        print(f"Generating {rack_type} break visualization...")
        print(f"{'='*60}")
        visualize_break_comparison(rack_type, break_speed, spacing_mode, seed)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'all':
            speed = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
            visualize_all_breaks(speed)
        else:
            rack_type = sys.argv[1]
            speed = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
            visualize_break_comparison(rack_type, speed)
    else:
        visualize_break_comparison()
