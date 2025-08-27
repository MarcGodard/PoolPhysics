#!/usr/bin/env python3
"""
Visualize the pool ball rack positions
"""
import numpy as np
import matplotlib.pyplot as plt
from pool_physics.table import PoolTable
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.table_renderer import draw_pool_table, draw_balls

def visualize_rack_positions(rack_type='8-ball'):
    """Create a visualization of the pool ball rack positions"""
    
    # Create a pool table instance
    table = PoolTable()
    
    # Get the racked positions
    positions = table.calc_racked_positions(rack_type=rack_type)
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot table outline
    table_width = table.W
    table_length = table.L
    
    # Draw table using unified renderer (standard style - the original reference)
    style_info = draw_pool_table(ax, table, style='standard', show_pockets=True, show_spots=True, show_grid=True)
    
    # Colors for balls and determine which balls to show
    ball_radius = table.ball_radius
    colors = plt.cm.Set3(np.linspace(0, 1, 16))  # Different colors for each ball
    
    if rack_type == '9-ball':
        balls_to_show = list(range(10))  # 0-9
    elif rack_type == '10-ball':
        balls_to_show = list(range(11))  # 0-10
    else:  # 8-ball
        balls_to_show = list(range(16))  # 0-15
    
    # Draw balls using unified renderer
    draw_balls(ax, positions, balls_to_show, ball_radius, colors, style_info, rack_type)
    
    # Set labels and title (unified function handled most formatting)
    ax.set_xlabel('X Position (meters)')
    ax.set_ylabel('Z Position (meters)')
    ax.set_title(f'{rack_type.title()} Rack Positions with Pockets\n(Ball 0 = Cue Ball, Red = Corner Pockets, Blue = Side Pockets)')
    
    # Add legend for spots and center line
    ax.plot([], [], 'r-', alpha=0.5, label='Center line')  # Empty plot for legend
    ax.plot([], [], 'ro', markersize=6, alpha=0.7, label='Foot spot (rack center)')
    ax.plot([], [], 'bo', markersize=6, alpha=0.7, label='Head spot (cue ball)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the visualization
    import os
    os.makedirs('images', exist_ok=True)
    filename = f'images/{rack_type}_rack_positions.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"{rack_type.title()} rack visualization saved as '{filename}'")
    
    # Print some statistics
    print(f"\nRack Statistics:")
    print(f"Rack type: {rack_type}")
    print(f"Table dimensions: {table_width:.3f}m × {table_length:.3f}m")
    print(f"Ball radius: {ball_radius:.4f}m")
    print(f"Ball diameter: {table.ball_diameter:.4f}m")
    
    print(f"\nCue ball position (ball 0): x={positions[0,0]:.3f}, z={positions[0,2]:.3f}")
    print(f"Rack apex (ball 1): x={positions[1,0]:.3f}, z={positions[1,2]:.3f}")
    
    plt.show()

def visualize_all_racks():
    """Visualize all three rack types"""
    for rack_type in ['8-ball', '9-ball', '10-ball']:
        print(f"\n{'='*50}")
        print(f"Generating {rack_type} rack visualization...")
        print(f"{'='*50}")
        visualize_rack_positions(rack_type)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        rack_type = sys.argv[1]
        if rack_type == 'all':
            visualize_all_racks()
        else:
            visualize_rack_positions(rack_type)
    else:
        visualize_rack_positions()
