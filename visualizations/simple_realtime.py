#!/usr/bin/env python3
"""
Simple real-time pool visualizer using matplotlib
Ready to use without OpenGL dependencies
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pool_physics import PoolPhysics
from pool_physics.table import PoolTable


class SimplePoolVisualizer:
    """Simple matplotlib-based pool visualizer"""
    
    def __init__(self, physics_engine: PoolPhysics):
        self.physics = physics_engine
        self.table = physics_engine.table
        
        # Create figure with better styling
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('darkgreen')
        
        # Set up the view
        margin = 0.3
        self.ax.set_xlim(-self.table.W/2 - margin, self.table.W/2 + margin)
        self.ax.set_ylim(-self.table.L/2 - margin, self.table.L/2 + margin)
        self.ax.set_aspect('equal')
        
        # Remove axes for cleaner look
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Animation state
        self.current_time = 0.0
        self.max_time = 0.0
        self.show_trails = True
        self.trail_history = {}
        self.trail_length = 50
        
        # Ball colors (professional pool ball colors)
        self.ball_colors = [
            '#FFFFFF',  # 0: Cue ball (white)
            '#FFFF00',  # 1: Yellow (solid)
            '#0000FF',  # 2: Blue (solid)
            '#FF0000',  # 3: Red (solid)
            '#800080',  # 4: Purple (solid)
            '#FFA500',  # 5: Orange (solid)
            '#008000',  # 6: Green (solid)
            '#800000',  # 7: Maroon (solid)
            '#000000',  # 8: Black (8-ball)
            '#FFFF00',  # 9: Yellow stripe
            '#0000FF',  # 10: Blue stripe
            '#FF0000',  # 11: Red stripe
            '#800080',  # 12: Purple stripe
            '#FFA500',  # 13: Orange stripe
            '#008000',  # 14: Green stripe
            '#800000',  # 15: Maroon stripe
        ]
        
        self._draw_table()
    
    def _draw_table(self):
        """Draw the pool table"""
        # Table surface (felt)
        table_rect = Rectangle((-self.table.W/2, -self.table.L/2), 
                              self.table.W, self.table.L, 
                              linewidth=4, edgecolor='#8B4513', 
                              facecolor='#006400', alpha=0.9)
        self.ax.add_patch(table_rect)
        
        # Rails
        rail_width = 0.05
        rail_color = '#8B4513'  # Saddle brown
        
        # Bottom rail
        self.ax.add_patch(Rectangle((-self.table.W/2-rail_width, -self.table.L/2-rail_width), 
                                   self.table.W+2*rail_width, rail_width, 
                                   facecolor=rail_color, edgecolor='#654321'))
        # Top rail
        self.ax.add_patch(Rectangle((-self.table.W/2-rail_width, self.table.L/2), 
                                   self.table.W+2*rail_width, rail_width, 
                                   facecolor=rail_color, edgecolor='#654321'))
        # Left rail
        self.ax.add_patch(Rectangle((-self.table.W/2-rail_width, -self.table.L/2), 
                                   rail_width, self.table.L, 
                                   facecolor=rail_color, edgecolor='#654321'))
        # Right rail
        self.ax.add_patch(Rectangle((self.table.W/2, -self.table.L/2), 
                                   rail_width, self.table.L, 
                                   facecolor=rail_color, edgecolor='#654321'))
        
        # Pockets
        pocket_radius = 0.07
        pocket_positions = [
            (-self.table.W/2, -self.table.L/2),  # Bottom left corner
            (self.table.W/2, -self.table.L/2),   # Bottom right corner
            (self.table.W/2, 0),                 # Right side
            (self.table.W/2, self.table.L/2),    # Top right corner
            (-self.table.W/2, self.table.L/2),   # Top left corner
            (-self.table.W/2, 0),                # Left side
        ]
        
        for px, py in pocket_positions:
            pocket = Circle((px, py), pocket_radius, color='black', zorder=10)
            self.ax.add_patch(pocket)
        
        # Table markings
        # Head string (breaking line)
        head_string_z = self.table.L/4
        self.ax.plot([-self.table.W/4, self.table.W/4], [head_string_z, head_string_z], 
                    'w--', alpha=0.5, linewidth=1)
        
        # Foot spot
        foot_spot_z = -self.table.L/4
        self.ax.plot(0, foot_spot_z, 'wo', markersize=4, alpha=0.7)
        
        # Head spot
        self.ax.plot(0, head_string_z, 'wo', markersize=4, alpha=0.7)
    
    def animate_shot(self, events, speed=1.0, save_animation=False, filename=None):
        """
        Animate a pool shot
        
        Args:
            events: Physics events to animate
            speed: Animation speed multiplier
            save_animation: Whether to save as video/gif
            filename: Output filename for saved animation
        """
        if not events:
            print("No events to animate")
            return None
        
        # Calculate animation parameters
        self.max_time = max(event.t for event in events if hasattr(event, 't'))
        fps = 30
        duration_seconds = self.max_time / speed
        total_frames = int(fps * duration_seconds)
        
        print(f"Animating {len(events)} events over {self.max_time:.2f}s physics time")
        print(f"Animation: {duration_seconds:.1f}s real time, {total_frames} frames")
        
        # Clear trail history
        self.trail_history.clear()
        
        def update_frame(frame):
            # Calculate current physics time
            self.current_time = (frame / total_frames) * self.max_time
            
            # Clear previous frame
            self.ax.clear()
            self._draw_table()
            
            # Get current ball positions
            try:
                positions = self.physics.eval_positions(self.current_time)
                velocities = self.physics.eval_velocities(self.current_time)
                
                # Update trails
                if self.show_trails:
                    self._update_trails(positions)
                    self._draw_trails()
                
                # Draw balls
                for i, pos in enumerate(positions):
                    if i < len(self.physics.balls_on_table) and self.physics._on_table[i]:
                        self._draw_ball(pos, i, velocities[i] if i < len(velocities) else None)
                
                # Add time display
                self.ax.text(0.02, 0.98, f'Time: {self.current_time:.2f}s', 
                            transform=self.ax.transAxes, color='white', 
                            fontsize=12, verticalalignment='top',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
                
                # Add ball speed info
                if len(velocities) > 0:
                    cue_speed = np.linalg.norm(velocities[0][:2]) if len(velocities[0]) >= 2 else 0
                    self.ax.text(0.02, 0.90, f'Cue Ball: {cue_speed:.1f} m/s', 
                                transform=self.ax.transAxes, color='yellow', 
                                fontsize=10, verticalalignment='top',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
                
            except Exception as e:
                print(f"Frame {frame}: Error getting positions at t={self.current_time:.3f}: {e}")
            
            self.ax.set_title('Pool Physics Real-time Simulation', color='white', fontsize=16, pad=20)
            
            return []
        
        # Create and run animation
        ani = animation.FuncAnimation(self.fig, update_frame, frames=total_frames, 
                                    interval=1000/fps, blit=False, repeat=False)
        
        # Save animation if requested
        if save_animation and filename:
            print(f"Saving animation to {filename}...")
            if filename.endswith('.gif'):
                ani.save(filename, writer='pillow', fps=fps//2)  # Slower fps for GIF
            elif filename.endswith('.mp4'):
                ani.save(filename, writer='ffmpeg', fps=fps)
            print(f"Animation saved: {filename}")
        
        # Show animation
        plt.tight_layout()
        plt.show()
        
        return ani
    
    def _draw_ball(self, position, ball_id, velocity=None):
        """Draw a single ball"""
        x, z = position[0], position[2]  # Use X-Z plane (top-down view)
        
        # Get ball color
        color = self.ball_colors[ball_id] if ball_id < len(self.ball_colors) else '#CCCCCC'
        
        # Draw ball shadow first
        shadow_offset = 0.003
        shadow = Circle((x + shadow_offset, z - shadow_offset), 
                       self.physics.ball_radius * 0.95, 
                       color='black', alpha=0.3, zorder=1)
        self.ax.add_patch(shadow)
        
        # Draw main ball
        ball = Circle((x, z), self.physics.ball_radius, 
                     color=color, edgecolor='white', linewidth=1, zorder=5)
        self.ax.add_patch(ball)
        
        # Add ball number
        if ball_id > 0:  # Don't number the cue ball
            text_color = 'white' if ball_id == 8 else 'black'
            self.ax.text(x, z, str(ball_id), fontsize=8, color=text_color, 
                        ha='center', va='center', weight='bold', zorder=6)
        
        # Show velocity vector
        if velocity is not None and np.linalg.norm(velocity[:2]) > 0.1:
            vx, vz = velocity[0], velocity[2]
            arrow_scale = 0.05
            self.ax.arrow(x, z, vx * arrow_scale, vz * arrow_scale,
                         head_width=self.physics.ball_radius * 0.3,
                         head_length=self.physics.ball_radius * 0.2,
                         fc='cyan', ec='cyan', alpha=0.8, zorder=7)
    
    def _update_trails(self, positions):
        """Update ball position trails"""
        for i, pos in enumerate(positions):
            if i < len(self.physics.balls_on_table) and self.physics._on_table[i]:
                if i not in self.trail_history:
                    self.trail_history[i] = []
                
                self.trail_history[i].append((pos[0], pos[2]))  # Store X-Z coordinates
                
                # Limit trail length
                if len(self.trail_history[i]) > self.trail_length:
                    self.trail_history[i].pop(0)
    
    def _draw_trails(self):
        """Draw ball movement trails"""
        for ball_id, trail in self.trail_history.items():
            if len(trail) < 2:
                continue
            
            # Extract coordinates
            xs = [point[0] for point in trail]
            zs = [point[1] for point in trail]
            
            # Draw trail with fading alpha
            for i in range(1, len(trail)):
                alpha = (i / len(trail)) * 0.6  # Fade from 0 to 0.6
                self.ax.plot([xs[i-1], xs[i]], [zs[i-1], zs[i]], 
                           color='cyan', alpha=alpha, linewidth=2, zorder=2)


def demo_break_shot():
    """Demo break shot animation"""
    print("Pool Break Shot Visualization Demo")
    print("=" * 40)
    
    # Set up physics
    physics = PoolPhysics()
    table = PoolTable()
    
    # Create 9-ball rack
    positions = table.calc_racked_positions('9-ball', spacing_mode='random', seed=42)
    physics.reset(ball_positions=positions, balls_on_table=list(range(10)))
    
    print("Simulating 9-ball break...")
    
    # Execute break shot
    events = physics.strike_ball(0.0, 0, 
                               positions[0],
                               positions[0] + np.array([0, 0, physics.ball_radius]),
                               np.array([0.05, 0.0, -14.0]), 0.54)
    
    print(f"Generated {len(events)} physics events")
    
    # Create visualizer and animate
    visualizer = SimplePoolVisualizer(physics)
    
    print("\nStarting animation...")
    print("Close the window to exit when animation completes.")
    
    # Animate the break shot
    ani = visualizer.animate_shot(events, speed=0.8)  # Slightly slower for better viewing
    
    return ani


def demo_custom_shot():
    """Demo custom shot setup"""
    print("Custom Shot Demo")
    print("=" * 20)
    
    # Set up physics
    physics = PoolPhysics()
    table = PoolTable()
    
    # Custom ball positions
    positions = np.zeros((16, 3))
    positions[:, 1] = table.H + physics.ball_radius  # Set height
    
    # Cue ball
    positions[0] = [0.0, table.H + physics.ball_radius, 0.5]
    # Object balls
    positions[1] = [0.1, table.H + physics.ball_radius, -0.2]
    positions[2] = [-0.1, table.H + physics.ball_radius, -0.2]
    positions[8] = [0.0, table.H + physics.ball_radius, -0.4]  # 8-ball
    
    physics.reset(ball_positions=positions, balls_on_table=[0, 1, 2, 8])
    
    print("Simulating custom shot...")
    
    # Execute shot
    events = physics.strike_ball(0.0, 0, 
                               positions[0],
                               positions[0] + np.array([0, 0, physics.ball_radius]),
                               np.array([0.2, 0.0, -3.0]), 0.54)
    
    print(f"Generated {len(events)} physics events")
    
    # Visualize
    visualizer = SimplePoolVisualizer(physics)
    ani = visualizer.animate_shot(events, speed=1.0)
    
    return ani


if __name__ == "__main__":
    # Run demo
    demo_break_shot()
    
    # Uncomment to try custom shot instead:
    # demo_custom_shot()