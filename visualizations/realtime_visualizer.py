#!/usr/bin/env python3
"""
Real-time pool table visualizer with OpenGL rendering

This module provides real-time visualization of pool physics simulations
with support for animated ball movement, collision effects, and future
projector alignment capabilities.

Usage:
    python realtime_visualizer.py
    
Features:
    - Real-time ball animation
    - Smooth physics interpolation
    - Ball trail effects
    - Collision highlighting
    - Multiple camera angles
    - Export capabilities for projector calibration
"""

import sys
import os
import time
import numpy as np
from typing import List, Tuple, Dict
import logging

# Core visualization dependencies
try:
    import pygame
    from pygame.locals import *
    from OpenGL.GL import *
    from OpenGL.GLU import *
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False
    print("OpenGL dependencies not found. Install with: pip install pygame PyOpenGL")

# Fallback to matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pool_physics import PoolPhysics
from pool_physics.table import PoolTable
from pool_physics.events import BallEvent

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


class PoolTableVisualizer:
    """
    Real-time pool table visualizer with multiple rendering backends
    """
    
    def __init__(self, 
                 physics_engine: PoolPhysics,
                 width: int = 1200, 
                 height: int = 800,
                 use_opengl: bool = True):
        """
        Initialize the visualizer
        
        Args:
            physics_engine: PoolPhysics instance
            width, height: Window dimensions
            use_opengl: Whether to use OpenGL (falls back to matplotlib if False)
        """
        self.physics = physics_engine
        self.table = physics_engine.table
        self.width = width
        self.height = height
        self.use_opengl = use_opengl and HAS_OPENGL
        
        # Animation state
        self.is_animating = False
        self.current_time = 0.0
        self.max_time = 0.0
        self.fps = 60
        self.speed_multiplier = 1.0
        
        # Visual settings
        self.show_trails = True
        self.trail_length = 30
        self.ball_trails: Dict[int, List[Tuple[float, float, float]]] = {}
        
        # Camera settings for projector mode
        self.projector_mode = False
        self.camera_height = 2.0
        self.camera_angle = 45.0  # degrees from horizontal
        
        # Initialize rendering backend
        if self.use_opengl:
            self._init_opengl()
        else:
            self._init_matplotlib()
    
    def _init_opengl(self):
        """Initialize OpenGL rendering context"""
        pygame.init()
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Pool Physics Real-time Visualizer")
        
        # OpenGL settings
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Set up lighting
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 2, 0, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
        
        # Set up camera
        self._setup_camera()
        
        _logger.info("OpenGL visualizer initialized")
    
    def _init_matplotlib(self):
        """Initialize matplotlib backend as fallback"""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-self.table.W/2 - 0.2, self.table.W/2 + 0.2)
        self.ax.set_ylim(-self.table.L/2 - 0.2, self.table.L/2 + 0.2)
        
        # Draw static table elements
        self._draw_table_matplotlib()
        
        _logger.info("Matplotlib visualizer initialized")
    
    def _setup_camera(self):
        """Set up OpenGL camera for optimal pool table viewing"""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        if self.projector_mode:
            # Orthographic projection for projector alignment
            margin = 0.2
            glOrtho(-self.table.W/2 - margin, self.table.W/2 + margin,
                   -self.table.L/2 - margin, self.table.L/2 + margin,
                   -1.0, 10.0)
            
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            # Top-down view
            gluLookAt(0, self.camera_height, 0,  # Camera position
                     0, 0, 0,                    # Look at table center
                     0, 0, 1)                    # Up vector
        else:
            # Perspective projection for better 3D visualization
            gluPerspective(45, self.width/self.height, 0.1, 50.0)
            
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            # Angled view of the table
            angle_rad = np.radians(self.camera_angle)
            camera_dist = 3.0
            camera_x = camera_dist * np.sin(angle_rad)
            camera_y = camera_dist * np.cos(angle_rad)
            
            gluLookAt(camera_x, camera_y, 0,  # Camera position
                     0, 0, 0,                 # Look at table center
                     0, 1, 0)                 # Up vector
    
    def animate_events(self, events: List[BallEvent], duration_override: float = None):
        """
        Animate a sequence of physics events
        
        Args:
            events: List of physics events to animate
            duration_override: Total animation duration (auto-calculated if None)
        """
        if not events:
            _logger.warning("No events to animate")
            return
        
        self.max_time = max(event.t for event in events if hasattr(event, 't'))
        if duration_override is None:
            duration = self.max_time / self.speed_multiplier
        else:
            duration = duration_override
        
        self.current_time = 0.0
        self.is_animating = True
        
        # Clear ball trails
        self.ball_trails.clear()
        
        _logger.info(f"Starting animation: {len(events)} events, {duration:.2f}s duration")
        
        if self.use_opengl:
            self._animate_opengl(duration)
        else:
            self._animate_matplotlib(duration)
    
    def _animate_opengl(self, duration: float):
        """Run OpenGL animation loop"""
        clock = pygame.time.Clock()
        running = True
        start_time = time.time()
        
        while running and self.is_animating:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_SPACE:
                        self.is_animating = not self.is_animating
                    elif event.key == K_r:
                        self.current_time = 0.0
                        start_time = time.time()
                    elif event.key == K_p:
                        self.projector_mode = not self.projector_mode
                        self._setup_camera()
                    elif event.key == K_t:
                        self.show_trails = not self.show_trails
            
            if self.is_animating:
                elapsed = time.time() - start_time
                self.current_time = elapsed * self.speed_multiplier
                
                if self.current_time >= self.max_time:
                    self.is_animating = False
                    _logger.info("Animation completed")
            
            self._render_opengl_frame()
            pygame.display.flip()
            clock.tick(self.fps)
        
        pygame.quit()
    
    def _render_opengl_frame(self):
        """Render a single OpenGL frame"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Draw table
        self._draw_table_opengl()
        
        # Get current ball positions
        positions = self.physics.eval_positions(self.current_time)
        
        # Update trails
        if self.show_trails:
            self._update_trails(positions)
        
        # Draw balls
        for i, pos in enumerate(positions):
            if i < len(self.physics._on_table) and self.physics._on_table[i]:
                self._draw_ball_opengl(pos, i)
        
        # Draw trails
        if self.show_trails:
            self._draw_trails_opengl()
    
    def _draw_table_opengl(self):
        """Draw pool table using OpenGL"""
        # Table surface (felt)
        glColor3f(0.0, 0.5, 0.0)  # Green felt
        glBegin(GL_QUADS)
        glVertex3f(-self.table.W/2, 0, -self.table.L/2)
        glVertex3f(self.table.W/2, 0, -self.table.L/2)
        glVertex3f(self.table.W/2, 0, self.table.L/2)
        glVertex3f(-self.table.W/2, 0, self.table.L/2)
        glEnd()
        
        # Table rails (simplified)
        glColor3f(0.4, 0.2, 0.1)  # Brown wood
        rail_height = 0.05
        rail_width = 0.05
        
        # Draw rails as simple rectangles
        self._draw_rail(-self.table.W/2 - rail_width, 0, -self.table.L/2 - rail_width,
                       self.table.W + 2*rail_width, rail_height, rail_width)  # Bottom
        self._draw_rail(-self.table.W/2 - rail_width, 0, self.table.L/2,
                       self.table.W + 2*rail_width, rail_height, rail_width)   # Top
        self._draw_rail(-self.table.W/2 - rail_width, 0, -self.table.L/2,
                       rail_width, rail_height, self.table.L)                 # Left
        self._draw_rail(self.table.W/2, 0, -self.table.L/2,
                       rail_width, rail_height, self.table.L)                 # Right
    
    def _draw_rail(self, x, y, z, width, height, length):
        """Draw a single rail segment"""
        glBegin(GL_QUADS)
        # Top face
        glVertex3f(x, y + height, z)
        glVertex3f(x + width, y + height, z)
        glVertex3f(x + width, y + height, z + length)
        glVertex3f(x, y + height, z + length)
        # Front face
        glVertex3f(x, y, z)
        glVertex3f(x + width, y, z)
        glVertex3f(x + width, y + height, z)
        glVertex3f(x, y + height, z)
        glEnd()
    
    def _draw_ball_opengl(self, position: np.ndarray, ball_id: int):
        """Draw a single ball using OpenGL"""
        glPushMatrix()
        glTranslatef(position[0], position[2], position[1])  # Note: Y and Z swapped for OpenGL
        
        # Ball colors
        colors = [
            (1.0, 1.0, 1.0),  # 0: Cue ball (white)
            (1.0, 1.0, 0.0),  # 1: Yellow
            (0.0, 0.0, 1.0),  # 2: Blue
            (1.0, 0.0, 0.0),  # 3: Red
            (0.5, 0.0, 0.5),  # 4: Purple
            (1.0, 0.5, 0.0),  # 5: Orange
            (0.0, 0.5, 0.0),  # 6: Green
            (0.5, 0.0, 0.0),  # 7: Maroon
            (0.0, 0.0, 0.0),  # 8: Black
            # Add more colors as needed
        ]
        
        if ball_id < len(colors):
            glColor3f(*colors[ball_id])
        else:
            glColor3f(0.8, 0.8, 0.8)  # Default gray
        
        # Draw sphere (using GLU quadric)
        quadric = gluNewQuadric()
        gluSphere(quadric, self.physics.ball_radius, 16, 16)
        gluDeleteQuadric(quadric)
        
        glPopMatrix()
    
    def _update_trails(self, positions: np.ndarray):
        """Update ball trails for smooth animation"""
        for i, pos in enumerate(positions):
            if i < len(self.physics._on_table) and self.physics._on_table[i]:
                if i not in self.ball_trails:
                    self.ball_trails[i] = []
                
                self.ball_trails[i].append((pos[0], pos[1], pos[2]))
                
                # Limit trail length
                if len(self.ball_trails[i]) > self.trail_length:
                    self.ball_trails[i].pop(0)
    
    def _draw_trails_opengl(self):
        """Draw ball trails using OpenGL"""
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        
        for ball_id, trail in self.ball_trails.items():
            if len(trail) < 2:
                continue
            
            glBegin(GL_LINE_STRIP)
            for i, (x, y, z) in enumerate(trail):
                # Fade trail over time
                alpha = i / len(trail)
                glColor4f(1.0, 1.0, 1.0, alpha * 0.5)
                glVertex3f(x, z, y)  # Note: Y and Z swapped
            glEnd()
        
        glEnable(GL_LIGHTING)
    
    def _animate_matplotlib(self, duration: float):
        """Fallback matplotlib animation"""
        def update_frame(frame):
            self.current_time = frame * duration / (self.fps * duration)
            if self.current_time > self.max_time:
                return []
            
            self.ax.clear()
            self._draw_table_matplotlib()
            
            # Get current positions
            positions = self.physics.eval_positions(self.current_time)
            
            # Draw balls
            for i, pos in enumerate(positions):
                if self.physics._on_table[i]:
                    circle = Circle((pos[0], pos[2]), self.physics.ball_radius, 
                                  color='white' if i == 0 else 'red', alpha=0.8)
                    self.ax.add_patch(circle)
            
            return []
        
        frames = int(self.fps * duration)
        ani = animation.FuncAnimation(self.fig, update_frame, frames=frames, 
                                    interval=1000/self.fps, blit=False, repeat=False)
        
        plt.show()
        return ani
    
    def _draw_table_matplotlib(self):
        """Draw table using matplotlib"""
        # Table outline
        table_rect = Rectangle((-self.table.W/2, -self.table.L/2), 
                              self.table.W, self.table.L, 
                              linewidth=3, edgecolor='brown', facecolor='green', alpha=0.8)
        self.ax.add_patch(table_rect)
        
        # Simplified pockets
        pocket_radius = 0.06
        pocket_positions = [
            (-self.table.W/2, -self.table.L/2),  # Bottom left
            (self.table.W/2, -self.table.L/2),   # Bottom right
            (self.table.W/2, 0),                 # Right side
            (self.table.W/2, self.table.L/2),    # Top right
            (-self.table.W/2, self.table.L/2),   # Top left
            (-self.table.W/2, 0),                # Left side
        ]
        
        for px, py in pocket_positions:
            pocket = Circle((px, py), pocket_radius, color='black')
            self.ax.add_patch(pocket)
        
        self.ax.set_title(f"Pool Physics Visualization (t={self.current_time:.2f}s)")
    
    def enable_projector_mode(self):
        """Enable projector alignment mode"""
        self.projector_mode = True
        if self.use_opengl:
            self._setup_camera()
        _logger.info("Projector mode enabled - orthographic top-down view")
    
    def save_calibration_frame(self, filename: str):
        """Save current frame for projector calibration"""
        if self.use_opengl:
            # Read OpenGL framebuffer
            glReadBuffer(GL_FRONT)
            pixels = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
            image = np.frombuffer(pixels, dtype=np.uint8).reshape(self.height, self.width, 3)
            image = np.flipud(image)  # Flip Y axis
            
            from PIL import Image
            img = Image.fromarray(image)
            img.save(filename)
        else:
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        
        _logger.info(f"Calibration frame saved: {filename}")


def demo_realtime_visualization():
    """Demo function showing real-time visualization capabilities"""
    print("Pool Physics Real-time Visualizer Demo")
    print("="*50)
    
    # Initialize physics
    physics = PoolPhysics()
    table = PoolTable()
    
    # Set up 9-ball break
    positions = table.calc_racked_positions('9-ball', spacing_mode='random', seed=42)
    physics.reset(ball_positions=positions, balls_on_table=list(range(10)))
    
    # Simulate break shot
    events = physics.strike_ball(0.0, 0, 
                               positions[0],
                               positions[0] + np.array([0, 0, physics.ball_radius]),
                               np.array([0.01, 0.0, -12.0]), 0.54)
    
    print(f"Simulated break shot: {len(events)} events generated")
    
    # Create visualizer
    visualizer = PoolTableVisualizer(physics, width=1400, height=900)
    
    print("\nStarting real-time animation...")
    print("Controls:")
    print("  SPACE - Pause/Resume")
    print("  R - Restart animation")
    print("  P - Toggle projector mode")
    print("  T - Toggle ball trails")
    print("  ESC/Close window - Exit")
    
    # Animate the shot
    visualizer.animate_events(events)


if __name__ == "__main__":
    demo_realtime_visualization()