#!/usr/bin/env python3
"""
Pool physics visualizer that exports high-quality videos
Perfect for when you can't see the live window
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pool_physics import PoolPhysics
from pool_physics.table import PoolTable


class VideoPoolVisualizer:
    """Export pool animations as video files"""
    
    def __init__(self, physics_engine: PoolPhysics, width=16, height=10, dpi=150):
        self.physics = physics_engine
        self.table = physics_engine.table
        
        # High-quality matplotlib setup
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('black')
        
        # Professional pool table colors
        self.ax.set_facecolor('#004d00')  # Dark green felt
        
        # Set view
        margin = 0.3
        self.ax.set_xlim(-self.table.W/2 - margin, self.table.W/2 + margin)
        self.ax.set_ylim(-self.table.L/2 - margin, self.table.L/2 + margin)
        self.ax.set_aspect('equal')
        
        # Remove axes for clean look
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Ball colors (professional)
        self.ball_colors = {
            0: '#FFFFFF',  # Cue ball
            1: '#FFFF00',  # 1-ball yellow
            2: '#0000FF',  # 2-ball blue  
            3: '#FF0000',  # 3-ball red
            4: '#800080',  # 4-ball purple
            5: '#FFA500',  # 5-ball orange
            6: '#008000',  # 6-ball green
            7: '#800000',  # 7-ball maroon
            8: '#000000',  # 8-ball black
            9: '#FFFF80',  # 9-ball yellow stripe
        }
        
        # Animation tracking
        self.trail_history = {}
        self.trail_length = 50
    
    def create_break_video(self, rack_type='9-ball', speed=14.0, filename=None, 
                          fps=30, duration=None, quality='high'):
        """Create a high-quality break shot video"""
        
        print(f"\n🎬 Creating {rack_type} break shot video...")
        print(f"   Break speed: {speed} m/s")
        
        # Set up physics
        positions = self.table.calc_racked_positions(rack_type, spacing_mode='random', seed=42)
        
        if rack_type == '9-ball':
            balls_on_table = list(range(10))
        elif rack_type == '10-ball':
            balls_on_table = list(range(11))
        else:
            balls_on_table = list(range(16))
            
        self.physics.reset(ball_positions=positions, balls_on_table=balls_on_table)
        
        # Simulate break
        events = self.physics.strike_ball(0.0, 0, 
                                        positions[0],
                                        positions[0] + np.array([0, 0, self.physics.ball_radius]),
                                        np.array([0.02, 0.0, -speed]), 0.54)
        
        print(f"   Generated {len(events)} physics events")
        
        # Calculate timing
        max_time = max(event.t for event in events if hasattr(event, 't'))
        if duration is None:
            duration = min(max_time, 15.0)  # Cap at 15 seconds
        
        total_frames = int(fps * duration)
        
        print(f"   Physics time: {max_time:.1f}s")
        print(f"   Video duration: {duration:.1f}s")
        print(f"   Total frames: {total_frames}")
        
        # Set filename
        if filename is None:
            filename = f"pool_{rack_type}_break_speed{speed:.0f}.mp4"
        
        # Clear trails
        self.trail_history.clear()
        
        # Animation function
        def animate_frame(frame):
            physics_time = (frame / total_frames) * max_time
            
            # Clear and redraw
            self.ax.clear()
            self._draw_professional_table()
            
            try:
                # Get ball states
                positions = self.physics.eval_positions(physics_time)
                velocities = self.physics.eval_velocities(physics_time)
                
                # Update trails
                self._update_trails(positions, physics_time)
                self._draw_trails()
                
                # Draw balls
                for i, pos in enumerate(positions):
                    if i < len(self.physics._on_table) and self.physics._on_table[i]:
                        velocity = velocities[i] if i < len(velocities) else None
                        self._draw_professional_ball(pos, i, velocity)
                
                # Add time and info
                self._add_info_overlay(physics_time, velocities)
                
            except Exception as e:
                print(f"Frame {frame}: Error at t={physics_time:.3f}: {e}")
            
            return []
        
        # Create animation
        print(f"🎬 Rendering {total_frames} frames...")
        
        anim = animation.FuncAnimation(self.fig, animate_frame, frames=total_frames,
                                     interval=1000/fps, blit=False, repeat=False)
        
        # Save video
        print(f"💾 Saving video: {filename}")
        
        if quality == 'high':
            writer_kwargs = {'bitrate': 5000, 'fps': fps}
        else:
            writer_kwargs = {'bitrate': 2000, 'fps': fps}
            
        anim.save(filename, writer='ffmpeg', **writer_kwargs, 
                 savefig_kwargs={'facecolor': 'black', 'edgecolor': 'none'})
        
        print(f"✅ Video saved: {filename}")
        print(f"   File size: {os.path.getsize(filename) / (1024*1024):.1f} MB")
        
        plt.close()
        return filename
    
    def _draw_professional_table(self):
        """Draw professional pool table"""
        # Table surface
        table_rect = Rectangle((-self.table.W/2, -self.table.L/2), 
                              self.table.W, self.table.L, 
                              linewidth=6, edgecolor='#8B4513', 
                              facecolor='#006600', alpha=0.9)
        self.ax.add_patch(table_rect)
        
        # Rails
        rail_width = 0.06
        rail_color = '#8B4513'
        
        # Draw rails
        rails = [
            # Bottom rail
            Rectangle((-self.table.W/2-rail_width, -self.table.L/2-rail_width), 
                     self.table.W+2*rail_width, rail_width, 
                     facecolor=rail_color, edgecolor='#654321', linewidth=2),
            # Top rail  
            Rectangle((-self.table.W/2-rail_width, self.table.L/2), 
                     self.table.W+2*rail_width, rail_width, 
                     facecolor=rail_color, edgecolor='#654321', linewidth=2),
            # Left rail
            Rectangle((-self.table.W/2-rail_width, -self.table.L/2), 
                     rail_width, self.table.L, 
                     facecolor=rail_color, edgecolor='#654321', linewidth=2),
            # Right rail
            Rectangle((self.table.W/2, -self.table.L/2), 
                     rail_width, self.table.L, 
                     facecolor=rail_color, edgecolor='#654321', linewidth=2),
        ]
        
        for rail in rails:
            self.ax.add_patch(rail)
        
        # Pockets
        pocket_radius = 0.07
        pocket_positions = [
            (-self.table.W/2, -self.table.L/2),  # Bottom left
            (self.table.W/2, -self.table.L/2),   # Bottom right
            (self.table.W/2, 0),                 # Right side
            (self.table.W/2, self.table.L/2),    # Top right
            (-self.table.W/2, self.table.L/2),   # Top left
            (-self.table.W/2, 0),                # Left side
        ]
        
        for px, py in pocket_positions:
            pocket = Circle((px, py), pocket_radius, color='black', zorder=10)
            self.ax.add_patch(pocket)
        
        # Table markings
        # Head string
        head_z = self.table.L/4
        self.ax.plot([-self.table.W/4, self.table.W/4], [head_z, head_z], 
                    'w--', alpha=0.6, linewidth=2)
        
        # Spots
        foot_spot = Circle((0, -self.table.L/4), 0.01, color='white', alpha=0.8)
        head_spot = Circle((0, head_z), 0.01, color='white', alpha=0.8)
        self.ax.add_patch(foot_spot)
        self.ax.add_patch(head_spot)
    
    def _draw_professional_ball(self, position, ball_id, velocity=None):
        """Draw high-quality ball"""
        x, z = position[0], position[2]
        
        # Ball color
        color = self.ball_colors.get(ball_id, '#CCCCCC')
        
        # Shadow
        shadow = Circle((x + 0.005, z - 0.005), 
                       self.physics.ball_radius * 0.95, 
                       color='black', alpha=0.4, zorder=1)
        self.ax.add_patch(shadow)
        
        # Main ball
        ball = Circle((x, z), self.physics.ball_radius, 
                     color=color, edgecolor='white', linewidth=2, zorder=5)
        self.ax.add_patch(ball)
        
        # Ball number
        if ball_id > 0:
            text_color = 'white' if ball_id == 8 else 'black'
            self.ax.text(x, z, str(ball_id), fontsize=12, color=text_color, 
                        ha='center', va='center', weight='bold', zorder=6)
        
        # Velocity arrow
        if velocity is not None:
            speed = np.linalg.norm(velocity[:2])
            if speed > 0.2:
                vx, vz = velocity[0], velocity[2]
                scale = 0.08
                length = min(speed * scale, 0.25)
                
                self.ax.arrow(x, z, vx * scale, vz * scale,
                             head_width=self.physics.ball_radius * 0.4,
                             head_length=self.physics.ball_radius * 0.3,
                             fc='cyan', ec='cyan', alpha=0.9, zorder=7, linewidth=2)
    
    def _update_trails(self, positions, physics_time):
        """Update ball trails"""
        for i, pos in enumerate(positions):
            if i < len(self.physics._on_table) and self.physics._on_table[i]:
                if i not in self.trail_history:
                    self.trail_history[i] = []
                
                self.trail_history[i].append((pos[0], pos[2], physics_time))
                
                # Limit trail length
                if len(self.trail_history[i]) > self.trail_length:
                    self.trail_history[i].pop(0)
    
    def _draw_trails(self):
        """Draw ball trails"""
        for ball_id, trail in self.trail_history.items():
            if len(trail) < 2:
                continue
            
            xs = [point[0] for point in trail]
            zs = [point[1] for point in trail]
            
            # Draw trail with fade
            for i in range(1, len(trail)):
                alpha = (i / len(trail)) * 0.7
                self.ax.plot([xs[i-1], xs[i]], [zs[i-1], zs[i]], 
                           color='cyan', alpha=alpha, linewidth=3, zorder=3)
    
    def _add_info_overlay(self, physics_time, velocities):
        """Add information overlay"""
        # Time display
        self.ax.text(0.02, 0.98, f'Time: {physics_time:.2f}s', 
                    transform=self.ax.transAxes, color='white', 
                    fontsize=16, verticalalignment='top', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.8))
        
        # Cue ball speed
        if len(velocities) > 0:
            cue_speed = np.linalg.norm(velocities[0][:2])
            self.ax.text(0.02, 0.90, f'Cue Ball: {cue_speed:.1f} m/s', 
                        transform=self.ax.transAxes, color='yellow', 
                        fontsize=14, verticalalignment='top', weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))
        
        # Title
        self.ax.set_title('🎱 Professional Pool Physics Simulation 🎱', 
                         color='white', fontsize=20, pad=20, weight='bold')


def main():
    """Main function for video creation"""
    parser = argparse.ArgumentParser(description='Create pool physics videos')
    parser.add_argument('--rack', choices=['8-ball', '9-ball', '10-ball'], 
                       default='9-ball', help='Rack type')
    parser.add_argument('--speed', type=float, default=14.0, help='Break speed')
    parser.add_argument('--output', type=str, help='Output filename')
    parser.add_argument('--fps', type=int, default=30, help='Video FPS')
    parser.add_argument('--duration', type=float, help='Video duration (auto if not set)')
    parser.add_argument('--quality', choices=['high', 'medium'], default='high')
    
    args = parser.parse_args()
    
    print("🎱 Pool Physics Video Creator 🎱")
    print("=" * 40)
    
    # Initialize physics
    physics = PoolPhysics()
    
    # Create visualizer
    visualizer = VideoPoolVisualizer(physics)
    
    # Create video
    filename = visualizer.create_break_video(
        rack_type=args.rack,
        speed=args.speed,
        filename=args.output,
        fps=args.fps,
        duration=args.duration,
        quality=args.quality
    )
    
    print(f"\n✅ Video creation complete!")
    print(f"📁 Output file: {filename}")
    print(f"🎬 Ready to view!")


if __name__ == "__main__":
    main()