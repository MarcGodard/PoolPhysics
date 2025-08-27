#!/usr/bin/env python3
"""
Visualize the pool ball rack positions
"""
import numpy as np
import matplotlib.pyplot as plt
from pool_physics.table import PoolTable

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
    
    # Draw table boundaries
    ax.plot([-table_width/2, table_width/2, table_width/2, -table_width/2, -table_width/2],
            [-table_length/2, -table_length/2, table_length/2, table_length/2, -table_length/2],
            'k-', linewidth=2, label='Table boundary')
    
    # Plot balls
    ball_radius = table.ball_radius
    colors = plt.cm.Set3(np.linspace(0, 1, 16))  # Different colors for each ball
    
    # Determine which balls to show based on rack type
    if rack_type == '9-ball':
        balls_to_show = list(range(10))  # 0-9
    elif rack_type == '10-ball':
        balls_to_show = list(range(11))  # 0-10
    else:  # 8-ball
        balls_to_show = list(range(16))  # 0-15
    
    for i in balls_to_show:
        x, y, z = positions[i]
        # Only plot if ball has a position (not at origin for unused balls)
        if rack_type != '8-ball' and i > 0 and x == 0 and z == 0:
            continue
            
        circle = plt.Circle((x, z), ball_radius, color=colors[i], alpha=0.7, edgecolor='black')
        ax.add_patch(circle)
        
        # Add ball number
        ax.text(x, z, str(i), ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Add labels and formatting
    ax.set_xlim(-table_width/2 - 0.1, table_width/2 + 0.1)
    ax.set_ylim(-table_length/2 - 0.1, table_length/2 + 0.1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position (meters)')
    ax.set_ylabel('Z Position (meters)')
    ax.set_title(f'{rack_type.title()} Rack Positions\n(Ball 0 = Cue Ball)')
    
    # Add some reference lines
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Center line')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add foot spot and head spot markers
    ax.plot(0, -table_length/4, 'ro', markersize=8, label='Foot spot (rack center)')
    ax.plot(0, table_length/4, 'bo', markersize=8, label='Head spot (cue ball)')
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the visualization
    filename = f'{rack_type}_rack_positions.png'
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
