#!/usr/bin/env python3
"""
Visualize pool break shots as time-series snapshots
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pool_physics.table import PoolTable
from pool_physics import PoolPhysics
from pool_physics.events import BallSlidingEvent
from table_renderer import draw_pool_table, draw_balls, draw_pocketed_balls

def simulate_break_timelapse(rack_type='8-ball', break_speed=12.0, max_time=6.0, time_interval=0.25):
    """
    Simulate a break shot and capture positions at regular intervals
    
    Args:
        rack_type: '8-ball', '9-ball', or '10-ball'
        break_speed: cue ball speed in m/s
        max_time: maximum simulation time in seconds
        time_interval: time interval between snapshots in seconds
    """
    print(f"\nSimulating {rack_type} break timelapse at {break_speed} m/s...")
    
    # Initialize physics
    physics = PoolPhysics()
    table = PoolTable()
    
    # Get rack positions
    initial_positions = table.calc_racked_positions(rack_type=rack_type)
    
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
    target_pos = initial_positions[1].copy()  # Ball 1 at foot spot
    
    # Calculate direction vector from cue ball to target
    direction = target_pos - cue_ball_pos
    direction[1] = 0  # Keep y=0 for horizontal shot
    direction_norm = np.linalg.norm(direction)
    direction_unit = direction / direction_norm
    
    # Set up cue ball velocity (aimed at apex ball)
    v_0 = direction_unit * break_speed
    omega_0 = np.array([0.0, -5.0, 0.0])  # Slight backspin initially
    
    # Create break event
    break_event = BallSlidingEvent(
        t=0.0,
        i=0,
        r_0=cue_ball_pos,
        v_0=v_0,
        omega_0=omega_0
    )
    
    print("Running break simulation...")
    events = physics.add_event_sequence(break_event)
    
    # Capture positions at specified intervals
    time_snapshots = []
    positions_snapshots = []
    
    for t in np.arange(0, max_time + time_interval, time_interval):
        positions = physics.eval_positions(t)
        time_snapshots.append(t)
        positions_snapshots.append(positions.copy())
    
    print(f"Generated {len(events)} events")
    print(f"Captured {len(time_snapshots)} time snapshots")
    
    return physics, time_snapshots, positions_snapshots, events, balls_on_table

def visualize_break_timelapse(rack_type='8-ball', break_speed=12.0, max_time=6.0, time_interval=0.25):
    """Create time-series visualization of break shot"""
    
    # Run simulation
    physics, time_snapshots, positions_snapshots, events, balls_on_table = simulate_break_timelapse(
        rack_type, break_speed, max_time, time_interval)
    table = PoolTable()
    
    # Calculate grid layout (adjust for higher resolution)
    n_snapshots = len(time_snapshots)
    n_cols = min(6, n_snapshots)  # Increased to 6 columns for more snapshots
    n_rows = (n_snapshots + n_cols - 1) // n_cols
    
    # Create figure with subplots (optimized for pool table aspect ratio)
    plt.close('all')
    # Pool table aspect ratio is W:L = 1:2, so panels should be wider than tall
    panel_width = 3.0  # Reduced from 4.0 to match table proportions
    panel_height = 4.5  # Increased slightly to maintain readability
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_width*n_cols, panel_height*n_rows))
    fig.set_dpi(150)  # High figure DPI for sharp display
    
    # Ensure axes is always 2D array
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Table dimensions
    table_width = table.W
    table_length = table.L
    ball_radius = table.ball_radius
    
    # Colors for balls
    colors = plt.cm.Set3(np.linspace(0, 1, 16))
    
    # Plot each time snapshot
    for idx, (t, positions) in enumerate(zip(time_snapshots, positions_snapshots)):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Draw table using unified renderer (standard style to match rack/break)
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
        
        # Set title with larger font for readable panels
        ax.set_title(f't = {t:.2f}s', fontsize=12, fontweight='bold')
        
        # Remove axis labels for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for idx in range(n_snapshots, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    # Add overall title with larger font for bigger figure
    fig.suptitle(f'{rack_type.title()} Break Timelapse - {break_speed} m/s', 
                 fontsize=18, fontweight='bold')
    
    # Optimize layout for larger readable panels
    plt.tight_layout(pad=1.0, h_pad=0.5, w_pad=0.5)
    
    # Save visualization with high DPI and optimized padding
    import os
    os.makedirs('images', exist_ok=True)
    filename = f'images/{rack_type}_break_timelapse.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    print(f"Break timelapse saved as '{filename}'")
    
    # Print break statistics
    print(f"\nTimelapse Statistics:")
    print(f"Rack type: {rack_type}")
    print(f"Break speed: {break_speed} m/s")
    print(f"Time snapshots: {len(time_snapshots)} (0-{max_time}s at {time_interval}s intervals)")
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
    
    plt.show()
    
    return physics, events

def visualize_all_break_timelapses(break_speed=12.0, max_time=6.0, time_interval=0.25):
    """Visualize break timelapse for all rack types"""
    for rack_type in ['8-ball', '9-ball', '10-ball']:
        print(f"\n{'='*60}")
        print(f"Generating {rack_type} break timelapse...")
        print(f"{'='*60}")
        visualize_break_timelapse(rack_type, break_speed, max_time, time_interval)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'all':
            speed = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
            max_time = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
            time_interval = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
            visualize_all_break_timelapses(speed, max_time, time_interval)
        else:
            rack_type = sys.argv[1]
            speed = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
            max_time = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
            time_interval = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
            visualize_break_timelapse(rack_type, speed, max_time, time_interval)
    else:
        visualize_break_timelapse()
