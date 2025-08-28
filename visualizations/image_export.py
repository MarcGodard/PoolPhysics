#!/usr/bin/env python3
"""Quick image export of pool shots"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pool_physics import PoolPhysics
from pool_physics.table import PoolTable

def create_break_images():
    """Create still images of break shot at different times"""
    
    # Initialize
    physics = PoolPhysics()
    table = PoolTable()
    positions = table.calc_racked_positions('9-ball', spacing_mode='random', seed=42)
    physics.reset(ball_positions=positions, balls_on_table=list(range(10)))
    
    # Simulate break
    events = physics.strike_ball(0.0, 0, positions[0],
                               positions[0] + np.array([0, 0, physics.ball_radius]),
                               np.array([0.02, 0.0, -14.0]), 0.54)
    
    max_time = max(event.t for event in events if hasattr(event, 't'))
    
    # Create images at different times
    times = [0.0, 0.1, 0.5, 1.0, 2.0]
    
    ball_colors = ['white', 'yellow', 'blue', 'red', 'purple', 'orange', 'green', 'maroon', 'black', 'gold']
    
    for i, t in enumerate(times):
        if t > max_time:
            continue
            
        plt.figure(figsize=(12, 8))
        plt.style.use('dark_background')
        ax = plt.gca()
        ax.set_facecolor('#004d00')
        
        # Draw table
        table_rect = Rectangle((-table.W/2, -table.L/2), table.W, table.L, 
                              linewidth=4, edgecolor='brown', facecolor='green', alpha=0.8)
        ax.add_patch(table_rect)
        
        # Pockets
        pocket_positions = [(-table.W/2, -table.L/2), (table.W/2, -table.L/2), (table.W/2, 0),
                           (table.W/2, table.L/2), (-table.W/2, table.L/2), (-table.W/2, 0)]
        for px, py in pocket_positions:
            pocket = Circle((px, py), 0.06, color='black')
            ax.add_patch(pocket)
        
        # Get positions at time t
        try:
            ball_positions = physics.eval_positions(t)
            velocities = physics.eval_velocities(t)
            
            # Draw balls
            for j, pos in enumerate(ball_positions):
                if j < len(physics._on_table) and physics._on_table[j]:
                    x, z = pos[0], pos[2]
                    color = ball_colors[j] if j < len(ball_colors) else 'gray'
                    
                    # Ball
                    ball = Circle((x, z), physics.ball_radius, color=color, 
                                 edgecolor='white', linewidth=1, zorder=5)
                    ax.add_patch(ball)
                    
                    # Number
                    if j > 0:
                        text_color = 'white' if j == 8 else 'black'
                        ax.text(x, z, str(j), fontsize=8, color=text_color, 
                               ha='center', va='center', weight='bold', zorder=6)
                    
                    # Velocity arrow
                    if j < len(velocities):
                        speed = np.linalg.norm(velocities[j][:2])
                        if speed > 0.1:
                            vx, vz = velocities[j][0], velocities[j][2]
                            scale = 0.05
                            ax.arrow(x, z, vx*scale, vz*scale, head_width=0.02, 
                                   head_length=0.015, fc='cyan', ec='cyan', alpha=0.8)
        
        except:
            pass
        
        ax.set_xlim(-table.W/2 - 0.2, table.W/2 + 0.2)
        ax.set_ylim(-table.L/2 - 0.2, table.L/2 + 0.2)
        ax.set_aspect('equal')
        ax.set_title(f'9-Ball Break Shot - Time: {t:.1f}s', color='white', fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        
        filename = f'break_shot_t{t:.1f}s.png'
        plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='black')
        plt.close()
        print(f'Created: {filename}')
    
    print(f'\\nCreated {len(times)} images showing the break shot progression')

if __name__ == "__main__":
    create_break_images()