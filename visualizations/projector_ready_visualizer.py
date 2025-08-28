#!/usr/bin/env python3
"""
Professional pool table visualizer optimized for projector alignment

This visualizer is specifically designed for projecting onto a real pool table,
with features like:
- Precise geometric calibration
- High contrast visuals for projection
- Real-time physics animation
- Keystone correction support
- Full-screen projection modes

Usage:
    python projector_ready_visualizer.py [--projector] [--fullscreen] [--calibrate]
"""

import sys
import os
import time
import argparse
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pool_physics import PoolPhysics
from pool_physics.table import PoolTable
from pool_physics.events import BallEvent

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


class ProjectorPoolVisualizer:
    """
    High-quality pool table visualizer designed for projector alignment
    """
    
    def __init__(self, 
                 physics_engine: PoolPhysics,
                 width: int = 1920,
                 height: int = 1080,
                 fullscreen: bool = False,
                 projector_mode: bool = True):
        
        self.physics = physics_engine
        self.table = physics_engine.table
        self.width = width
        self.height = height
        self.fullscreen = fullscreen
        self.projector_mode = projector_mode
        
        # High-quality animation settings
        self.fps = 60
        self.speed_multiplier = 1.0
        self.current_time = 0.0
        self.max_time = 0.0
        self.is_animating = False
        
        # Visual enhancement settings
        self.show_trails = True
        self.show_velocity_arrows = False
        self.show_prediction_paths = False
        self.high_contrast_mode = True
        self.trail_length = 40
        
        # Ball trail storage
        self.ball_trails = {}
        
        # Calibration grid for projector alignment
        self.show_calibration_grid = False
        self.grid_spacing = 0.1  # 10cm grid
        
        # Color schemes for different modes
        self.colors = self._init_color_schemes()
        
        # Initialize OpenGL
        self._init_opengl()
        
        _logger.info(f"Projector visualizer initialized: {width}x{height}, projector_mode={projector_mode}")
    
    def _init_color_schemes(self):
        """Initialize color schemes for different projection scenarios"""
        return {
            'standard': {
                'table': (0.0, 0.4, 0.0),      # Dark green felt
                'rails': (0.4, 0.2, 0.1),      # Brown wood
                'pockets': (0.0, 0.0, 0.0),    # Black
                'balls': {
                    0: (1.0, 1.0, 1.0),         # Cue ball - white
                    # Official American 8-ball solid colors
                    1: (1.0, 1.0, 0.1),         # 1-ball - bright yellow
                    2: (0.1, 0.1, 1.0),         # 2-ball - bright blue  
                    3: (1.0, 0.1, 0.1),         # 3-ball - bright red
                    4: (0.6, 0.1, 0.9),         # 4-ball - purple
                    5: (1.0, 0.6, 0.1),         # 5-ball - orange
                    6: (0.1, 0.8, 0.1),         # 6-ball - green
                    7: (0.6, 0.1, 0.1),         # 7-ball - maroon (dark red)
                    8: (0.1, 0.1, 0.1),         # 8-ball - black
                    # Striped balls (9-15) - same colors as solids for stripes
                    9: (1.0, 1.0, 0.1),         # 9-ball - yellow stripe
                    10: (0.1, 0.1, 1.0),        # 10-ball - blue stripe
                    11: (1.0, 0.1, 0.1),        # 11-ball - red stripe
                    12: (0.6, 0.1, 0.9),        # 12-ball - purple stripe
                    13: (1.0, 0.6, 0.1),        # 13-ball - orange stripe
                    14: (0.1, 0.8, 0.1),        # 14-ball - green stripe
                    15: (0.6, 0.1, 0.1),        # 15-ball - maroon stripe
                },
                'trails': (0.0, 1.0, 1.0),      # Cyan trails
                'grid': (0.3, 0.3, 0.3),        # Gray grid
            },
            'high_contrast': {
                'table': (0.0, 0.2, 0.0),      # Darker green for projection
                'rails': (0.6, 0.4, 0.2),      # Lighter brown for visibility
                'pockets': (0.1, 0.1, 0.1),    # Dark gray (not pure black)
                'balls': {
                    0: (1.0, 1.0, 1.0),         # Bright white
                    # High contrast solid balls for projection
                    1: (1.0, 1.0, 0.2),         # Bright yellow
                    2: (0.2, 0.2, 1.0),         # Bright blue
                    3: (1.0, 0.2, 0.2),         # Bright red
                    4: (0.8, 0.2, 1.0),         # Bright purple
                    5: (1.0, 0.7, 0.2),         # Bright orange
                    6: (0.2, 1.0, 0.2),         # Bright green
                    7: (0.8, 0.2, 0.2),         # Bright maroon
                    8: (0.4, 0.4, 0.4),         # Dark gray (visible on projection)
                    # Striped balls - same colors for projection visibility
                    9: (1.0, 1.0, 0.2),         # Bright yellow stripe
                    10: (0.2, 0.2, 1.0),        # Bright blue stripe
                    11: (1.0, 0.2, 0.2),        # Bright red stripe
                    12: (0.8, 0.2, 1.0),        # Bright purple stripe
                    13: (1.0, 0.7, 0.2),        # Bright orange stripe
                    14: (0.2, 1.0, 0.2),        # Bright green stripe
                    15: (0.8, 0.2, 0.2),        # Bright maroon stripe
                },
                'trails': (0.2, 1.0, 1.0),      # Bright cyan
                'grid': (0.5, 0.5, 0.5),        # Bright gray
            }
        }
    
    def _init_opengl(self):
        """Initialize high-quality OpenGL context"""
        pygame.init()
        
        # Set OpenGL attributes for better quality
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)  # 4x MSAA
        
        # Create display
        flags = DOUBLEBUF | OPENGL
        if self.fullscreen:
            flags |= FULLSCREEN
            # Get desktop resolution for fullscreen
            info = pygame.display.Info()
            self.width, self.height = info.current_w, info.current_h
        
        pygame.display.set_mode((self.width, self.height), flags)
        pygame.display.set_caption("Professional Pool Physics Projector - Press 'H' for Help")
        
        # High-quality OpenGL settings
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)  # Enable antialiasing
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        
        # Distributed lighting system for even table illumination
        glEnable(GL_LIGHTING)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Enable multiple lights for distributed coverage
        glEnable(GL_LIGHT0)  # Center overhead
        glEnable(GL_LIGHT1)  # Corner light 1
        glEnable(GL_LIGHT2)  # Corner light 2
        glEnable(GL_LIGHT3)  # Corner light 3
        
        # Main center light (overhead like pool hall lighting)
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 2.5, 0, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.6, 0.6, 0.6, 1])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.3, 0.3, 0.3, 1])
        
        # Corner light 1 (top-left area)
        glLightfv(GL_LIGHT1, GL_POSITION, [-self.table.W*0.6, 2.0, self.table.L*0.6, 1])
        glLightfv(GL_LIGHT1, GL_AMBIENT, [0.1, 0.1, 0.1, 1])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.4, 0.4, 0.4, 1])
        glLightfv(GL_LIGHT1, GL_SPECULAR, [0.2, 0.2, 0.2, 1])
        
        # Corner light 2 (top-right area)
        glLightfv(GL_LIGHT2, GL_POSITION, [self.table.W*0.6, 2.0, self.table.L*0.6, 1])
        glLightfv(GL_LIGHT2, GL_AMBIENT, [0.1, 0.1, 0.1, 1])
        glLightfv(GL_LIGHT2, GL_DIFFUSE, [0.4, 0.4, 0.4, 1])
        glLightfv(GL_LIGHT2, GL_SPECULAR, [0.2, 0.2, 0.2, 1])
        
        # Corner light 3 (bottom area)
        glLightfv(GL_LIGHT3, GL_POSITION, [0, 2.0, -self.table.L*0.6, 1])
        glLightfv(GL_LIGHT3, GL_AMBIENT, [0.1, 0.1, 0.1, 1])
        glLightfv(GL_LIGHT3, GL_DIFFUSE, [0.4, 0.4, 0.4, 1])
        glLightfv(GL_LIGHT3, GL_SPECULAR, [0.2, 0.2, 0.2, 1])
        
        # Enable specular highlights on materials
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.5, 0.5, 0.5, 1])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 16.0)
        
        # Set up camera
        self._setup_camera()
        
        # Set background color (black for projection)
        glClearColor(0.0, 0.0, 0.0, 1.0)
    
    def _setup_camera(self):
        """Set up optimal camera for projector or 3D viewing"""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        # 9 feet above the floor in meters (9 feet = 108 inches)
        CAMERA_HEIGHT_ABOVE_FLOOR = 108 * 0.0254  # 2.7432 meters
        
        if self.projector_mode:
            # Zoomed orthographic projection for better detail visibility
            # Smaller margin for closer view
            margin = 0.05  # Reduced from 0.15 for closer zoom
            table_w, table_l = self.table.W, self.table.L
            
            glOrtho(-table_w/2 - margin, table_w/2 + margin,
                   -table_l/2 - margin, table_l/2 + margin,
                   -1.0, 5.0)
            
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            
            # Top-down view from 9 feet above floor, rotated 90 degrees
            camera_y = CAMERA_HEIGHT_ABOVE_FLOOR - self.table.H  # Height above table surface
            gluLookAt(0, camera_y, 0,    # Camera position (9 feet above floor)
                     0, 0, 0,            # Look at table center
                     1, 0, 0)            # Up vector (X points forward - 90° rotation)
        else:
            # Zoomed top-down perspective view from 9 feet above floor
            aspect_ratio = self.width / self.height
            gluPerspective(45, aspect_ratio, 0.1, 10.0)  # Narrower FOV for closer zoom (was 60)
            
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            
            # Direct overhead view from 9 feet above floor, rotated 90 degrees
            camera_y = CAMERA_HEIGHT_ABOVE_FLOOR - self.table.H  # Height above table surface
            gluLookAt(0, camera_y, 0,    # Camera 9 feet above floor
                     0, 0, 0,            # Look at table center
                     1, 0, 0)            # Up vector (X points forward - 90° rotation)
    
    def animate_shot(self, events, speed=1.0, show_info=True):
        """
        Animate a pool shot with professional quality
        
        Args:
            events: Physics events to animate
            speed: Animation speed multiplier
            show_info: Whether to show information overlays
        """
        if not events:
            _logger.warning("No events to animate")
            return
        
        # Calculate timing
        self.max_time = max(event.t for event in events if hasattr(event, 't'))
        self.speed_multiplier = speed
        self.current_time = 0.0
        self.is_animating = True
        
        # Clear trails and reset ball rotations
        self.ball_trails.clear()
        if hasattr(self, '_ball_rotations'):
            self._ball_rotations.clear()
        
        _logger.info(f"Starting professional animation: {len(events)} events, {self.max_time:.2f}s physics time")
        
        # Main animation loop
        clock = pygame.time.Clock()
        start_time = time.time()
        show_help = False
        paused = False
        
        while True:
            # Handle events
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    pygame.quit()
                    return
                
                elif event.type == KEYDOWN:
                    if event.key == K_SPACE:
                        paused = not paused
                        if not paused:
                            start_time = time.time() - (self.current_time / self.speed_multiplier)
                    elif event.key == K_r:
                        # Restart animation
                        self.current_time = 0.0
                        start_time = time.time()
                        self.ball_trails.clear()
                        paused = False
                    elif event.key == K_h:
                        show_help = not show_help
                    elif event.key == K_t:
                        self.show_trails = not self.show_trails
                    elif event.key == K_v:
                        self.show_velocity_arrows = not self.show_velocity_arrows
                    elif event.key == K_g:
                        self.show_calibration_grid = not self.show_calibration_grid
                    elif event.key == K_c:
                        self.high_contrast_mode = not self.high_contrast_mode
                    elif event.key == K_PLUS or event.key == K_EQUALS:
                        self.speed_multiplier = min(5.0, self.speed_multiplier * 1.2)
                        start_time = time.time() - (self.current_time / self.speed_multiplier)
                    elif event.key == K_MINUS:
                        self.speed_multiplier = max(0.1, self.speed_multiplier / 1.2)
                        start_time = time.time() - (self.current_time / self.speed_multiplier)
            
            # Update animation time
            if not paused and self.current_time < self.max_time:
                elapsed = time.time() - start_time
                self.current_time = elapsed * self.speed_multiplier
            
            # Check if animation completed
            if self.current_time >= self.max_time and self.is_animating:
                self.is_animating = False
                _logger.info("Animation completed - press 'R' to restart")
            
            # Render frame
            self._render_frame(show_info, show_help)
            
            pygame.display.flip()
            clock.tick(self.fps)
    
    def _render_frame(self, show_info=True, show_help=False):
        """Render a single high-quality frame"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Select color scheme
        colors = self.colors['high_contrast' if self.high_contrast_mode else 'standard']
        
        # Draw calibration grid if enabled
        if self.show_calibration_grid:
            self._draw_calibration_grid(colors['grid'])
        
        # Draw table
        self._draw_table_professional(colors)
        
        # Get current state
        try:
            positions = self.physics.eval_positions(self.current_time)
            velocities = self.physics.eval_velocities(self.current_time)
            
            # Update trails
            if self.show_trails:
                self._update_trails(positions)
                self._draw_trails(colors['trails'])
            
            # Draw balls - realistic 3D approach with spin
            try:
                angular_velocities = self.physics.eval_angular_velocities(self.current_time)
            except:
                angular_velocities = [np.array([0, 0, 0]) for _ in positions]
            
            for i, pos in enumerate(positions):
                if i < len(self.physics._on_table) and self.physics._on_table[i]:
                    ball_color = colors['balls'].get(i, (0.8, 0.8, 0.8))
                    velocity = velocities[i] if i < len(velocities) else None
                    angular_vel = angular_velocities[i] if i < len(angular_velocities) else np.array([0, 0, 0])
                    self._draw_ball_professional(pos, ball_color, i, velocity, angular_vel)
                elif i == 0:  # Always draw cue ball, even if marked as off table
                    if i >= len(self.physics._on_table):
                        _logger.warning(f'Cue ball index {i} out of range for _on_table array (length: {len(self.physics._on_table)})')
                    # Suppress warning for cue ball being marked as off table - this is expected behavior
                    
                    # Always draw cue ball in proper white color
                    ball_color = (1.0, 1.0, 1.0)  # Pure white for cue ball
                    velocity = velocities[i] if i < len(velocities) else None
                    angular_vel = angular_velocities[i] if i < len(angular_velocities) else np.array([0, 0, 0])
                    self._draw_ball_professional(pos, ball_color, i, velocity, angular_vel)
        
        except Exception as e:
            _logger.warning(f"Error rendering at t={self.current_time:.3f}: {e}")
        
        # Draw UI overlays (only in non-projector mode or if explicitly enabled)
        if show_info and not self.projector_mode:
            self._draw_info_overlay()
        
        if show_help:
            self._draw_help_overlay()
    
    def _draw_table_professional(self, colors):
        """Draw simplified table for projection"""
        # No green background - just table outline and pockets
        
        # Draw table outline
        glColor3f(1.0, 1.0, 1.0)  # White outline
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        glVertex3f(-self.table.W/2, 0.001, -self.table.L/2)
        glVertex3f(self.table.W/2, 0.001, -self.table.L/2)
        glVertex3f(self.table.W/2, 0.001, self.table.L/2)
        glVertex3f(-self.table.W/2, 0.001, self.table.L/2)
        glEnd()
        glLineWidth(1.0)
        
        # Pockets
        self._draw_pockets()
        
        # Table markings for professional appearance
        self._draw_table_markings(colors)
    
    
    def _draw_pockets(self):
        """Draw simple pocket outlines based on WPA specs"""
        # WPA regulation pocket positions and dimensions
        corner_mouth = 0.1175 / 2  # 4.625" diameter = 2.3125" radius
        side_mouth = 0.130175 / 2  # 5.125" diameter = 2.5625" radius
        
        pocket_positions = [
            (-self.table.W/2, -self.table.L/2, 'corner'),  # Bottom left corner
            (self.table.W/2, -self.table.L/2, 'corner'),   # Bottom right corner
            (self.table.W/2, 0, 'side'),                   # Right side
            (self.table.W/2, self.table.L/2, 'corner'),    # Top right corner
            (-self.table.W/2, self.table.L/2, 'corner'),   # Top left corner
            (-self.table.W/2, 0, 'side'),                  # Left side
        ]
        
        # Draw simple lines for pocket outlines
        glColor3f(1.0, 1.0, 1.0)  # White lines
        glLineWidth(2.0)
        
        for px, pz, pocket_type in pocket_positions:
            radius = corner_mouth if pocket_type == 'corner' else side_mouth
            self._draw_pocket_outline(px, pz, radius)
        
        glLineWidth(1.0)
    
    def _draw_pocket_outline(self, x, z, radius):
        """Draw a simple circular pocket outline"""
        segments = 24
        
        # Draw circle outline for pocket
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            px = x + radius * np.cos(angle)
            pz = z + radius * np.sin(angle)
            glVertex3f(px, 0.001, pz)
        glEnd()
    
    def _draw_ball_professional(self, position, color, ball_id, velocity=None, angular_velocity=None):
        """Draw high-quality 3D ball with professional appearance and rotating stripes"""
        x, y, z = position[0], position[1] - self.table.H, position[2]
        
        glPushMatrix()
        glTranslatef(x, y, z)
        
        # Apply rotation based on angular velocity (if provided)
        if angular_velocity is not None:
            # For ball spin, we need to track cumulative rotation over time
            # Angular velocity is in rad/s, so we integrate over time
            if not hasattr(self, '_ball_rotations'):
                self._ball_rotations = {}
            
            if ball_id not in self._ball_rotations:
                self._ball_rotations[ball_id] = np.array([0.0, 0.0, 0.0])
            
            # Only update rotation if ball is actually spinning (avoid jumping when stopped)
            angular_speed = np.linalg.norm(angular_velocity)
            if angular_speed > 0.001:
                # Ball is spinning - integrate angular velocity
                dt = 1.0 / self.fps  # Time step
                self._ball_rotations[ball_id] += angular_velocity * dt
            # If angular_speed <= 0.001, ball has stopped - keep last rotation state
            
            # Always apply the current rotation state (whether updating or preserved)
            rotation_x = self._ball_rotations[ball_id][0] * 180 / np.pi
            rotation_y = self._ball_rotations[ball_id][1] * 180 / np.pi  
            rotation_z = self._ball_rotations[ball_id][2] * 180 / np.pi
            
            # Apply rotations in proper order (X, Y, Z)
            glRotatef(rotation_x, 1, 0, 0)
            glRotatef(rotation_y, 0, 1, 0)
            glRotatef(rotation_z, 0, 0, 1)
        
        # Determine if this is a striped ball (9-15)
        is_striped = ball_id >= 9 and ball_id <= 15
        
        if is_striped:
            # Draw striped ball with alternating colored and white sections
            glDisable(GL_LIGHTING)  # Temporarily disable lighting for strong colors
            self._draw_striped_sphere(color, ball_id)
            glEnable(GL_LIGHTING)
        else:
            # Draw solid colored ball with strong colors
            glDisable(GL_LIGHTING)  # Temporarily disable lighting for vibrant colors
            glColor3f(*color)
            sphere = gluNewQuadric()
            gluQuadricNormals(sphere, GLU_SMOOTH)
            gluSphere(sphere, self.physics.ball_radius, 24, 24)
            gluDeleteQuadric(sphere)
            glEnable(GL_LIGHTING)
        
        # Ball numbers removed - not readable enough for this view
        # if ball_id > 0:
        #     self._draw_ball_number(ball_id, is_striped)
        
        glPopMatrix()
        
        # Optional velocity arrow (controlled by show_velocity_arrows setting)
        if self.show_velocity_arrows and velocity is not None:
            speed = np.linalg.norm(velocity[:3])
            if speed > 0.1:  # Only show if ball is moving significantly
                self._draw_velocity_arrow(position, velocity, color)
    
    def _draw_striped_sphere(self, color, ball_id):
        """Draw a sphere with a single colored stripe around the middle (proper pool ball design)"""
        radius = self.physics.ball_radius
        segments = 24  # Longitude segments
        rings = 12     # Latitude segments
        
        # Draw sphere - white base with one colored stripe in the middle
        for i in range(rings):
            lat1 = np.pi * (-0.5 + float(i) / rings)
            lat2 = np.pi * (-0.5 + float(i + 1) / rings)
            
            # Determine if this latitude band is in the stripe zone
            # Adjusted stripe width: 20% thinner than 1/3 circumference for better realism
            center_lat = (lat1 + lat2) / 2
            stripe_width = (2.0 * np.pi / 3) * 0.8  # 80% of 1/3 circumference (10% more reduction)
            
            # Check if this ring is in the stripe zone (around equator)
            if abs(center_lat) < stripe_width / 2:
                stripe_color = color  # Colored stripe
            else:
                stripe_color = (1.0, 1.0, 1.0)  # White
            
            glColor3f(*stripe_color)
            
            glBegin(GL_QUAD_STRIP)
            for j in range(segments + 1):
                lng = 2 * np.pi * float(j) / segments
                
                # First vertex (current latitude ring)
                x1 = radius * np.cos(lat1) * np.cos(lng)
                y1 = radius * np.sin(lat1)
                z1 = radius * np.cos(lat1) * np.sin(lng)
                glVertex3f(x1, y1, z1)
                
                # Second vertex (next latitude ring)
                x2 = radius * np.cos(lat2) * np.cos(lng)
                y2 = radius * np.sin(lat2)
                z2 = radius * np.cos(lat2) * np.sin(lng)
                glVertex3f(x2, y2, z2)
            glEnd()
    
    def _draw_ball_number(self, ball_id, is_striped):
        """Draw ball number according to official specifications"""
        glPushMatrix()
        
        # Position number directly on the ball surface
        glTranslatef(0, self.physics.ball_radius * 0.95, 0)  # On the ball surface
        
        # Disable lighting for clear number visibility
        glDisable(GL_LIGHTING)
        
        # Use proper contrasting colors for pool balls
        if ball_id == 8:
            glColor3f(1.0, 1.0, 1.0)  # White on black 8-ball
        elif ball_id == 1:  # Yellow ball
            glColor3f(0.0, 0.0, 0.0)  # Black on yellow
        else:
            glColor3f(1.0, 1.0, 1.0)  # White on colored balls
        
        # Draw larger number for better readability
        self._draw_number_shape(ball_id, self.physics.ball_radius * 0.35)  # Increased from 0.25
        
        # Re-enable lighting
        glEnable(GL_LIGHTING)
        glPopMatrix()
    
    def _draw_number_shape(self, number, size):
        """Draw the actual ball number as a simple shape"""
        glLineWidth(4.0)
        
        # Draw a filled circle first as background
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, 0, 0)
        for i in range(17):
            angle = 2 * np.pi * i / 16
            x = size * 0.8 * np.cos(angle)
            z = size * 0.8 * np.sin(angle)
            glVertex3f(x, 0, z)
        glEnd()
        
        # Change color for the number itself (opposite contrast)
        if number == 8:
            glColor3f(0.0, 0.0, 0.0)  # Black number on white background
        elif number == 1:
            glColor3f(1.0, 1.0, 1.0)  # White number on black background  
        else:
            glColor3f(0.0, 0.0, 0.0)  # Black number on white background
        
        # Draw simple representation of the number using larger dots
        if number <= 9:
            # Draw larger dots in a pattern representing the number
            glPointSize(12.0)  # Increased from 8.0
            glBegin(GL_POINTS)
            for i in range(number):
                angle = 2 * np.pi * i / max(number, 3)  # Prevent division issues
                radius = size * 0.4
                x = radius * np.cos(angle)
                z = radius * np.sin(angle)
                glVertex3f(x, 0, z)
            glEnd()
            glPointSize(1.0)
        else:
            # For 10-15, draw two concentric rings of larger dots
            glPointSize(10.0)  # Increased from 6.0
            glBegin(GL_POINTS)
            # Inner ring (representing "1" in "1X")
            for i in range(5):
                angle = 2 * np.pi * i / 5
                x = size * 0.25 * np.cos(angle)
                z = size * 0.25 * np.sin(angle)
                glVertex3f(x, 0, z)
            # Outer ring (representing the second digit)
            remaining = number - 10
            for i in range(max(remaining, 1)):
                angle = 2 * np.pi * i / max(remaining, 1)
                x = size * 0.55 * np.cos(angle)
                z = size * 0.55 * np.sin(angle)
                glVertex3f(x, 0, z)
            glEnd()
            glPointSize(1.0)
        
        glLineWidth(1.0)
    
    def _draw_velocity_arrow(self, position, velocity, color):
        """Draw velocity vector as arrow"""
        x, y, z = position[0], position[1] - self.table.H, position[2]
        vx, vy, vz = velocity[0], 0, velocity[2]  # Only horizontal components
        
        # Scale arrow based on velocity
        scale = 0.1
        arrow_length = min(np.linalg.norm([vx, vz]) * scale, 0.3)  # Cap arrow length
        
        if arrow_length < 0.01:
            return
        
        # Normalize velocity for direction
        v_norm = np.linalg.norm([vx, vz])
        if v_norm > 0:
            vx_norm = vx / v_norm
            vz_norm = vz / v_norm
        else:
            return
        
        # Arrow color (brighter version of ball color)
        arrow_color = tuple(min(1.0, c + 0.3) for c in color)
        glColor3f(*arrow_color)
        
        # Draw arrow
        glLineWidth(3.0)
        glBegin(GL_LINES)
        # Shaft
        glVertex3f(x, y + 0.01, z)
        glVertex3f(x + vx_norm * arrow_length, y + 0.01, z + vz_norm * arrow_length)
        glEnd()
        
        # Arrow head (simplified)
        head_size = self.physics.ball_radius * 0.3
        glBegin(GL_TRIANGLES)
        ax, az = x + vx_norm * arrow_length, z + vz_norm * arrow_length
        # Simple triangle arrowhead
        glVertex3f(ax, y + 0.01, az)
        glVertex3f(ax - vx_norm * head_size - vz_norm * head_size * 0.5, y + 0.01, 
                   az - vz_norm * head_size + vx_norm * head_size * 0.5)
        glVertex3f(ax - vx_norm * head_size + vz_norm * head_size * 0.5, y + 0.01, 
                   az - vz_norm * head_size - vx_norm * head_size * 0.5)
        glEnd()
        
        glLineWidth(1.0)
    
    def _draw_table_markings(self, colors):
        """Draw professional table markings"""
        # Head string (break line)
        head_z = self.table.L/4
        foot_z = -self.table.L/4
        
        glColor3f(0.8, 0.8, 0.8)  # Light gray for markings
        glLineWidth(2.0)
        
        # Head string
        glBegin(GL_LINES)
        glVertex3f(-self.table.W/4, 0.002, head_z)
        glVertex3f(self.table.W/4, 0.002, head_z)
        glEnd()
        
        # Foot spot and head spot
        self._draw_spot(0, foot_z)  # Foot spot
        self._draw_spot(0, head_z)  # Head spot
        
        glLineWidth(1.0)
    
    def _draw_spot(self, x, z, radius=0.008):
        """Draw a table spot marker"""
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(x, 0.002, z)
        
        segments = 8
        for i in range(segments + 1):
            angle = 2 * np.pi * i / segments
            sx = x + radius * np.cos(angle)
            sz = z + radius * np.sin(angle)
            glVertex3f(sx, 0.002, sz)
        glEnd()
    
    
    def _draw_calibration_grid(self, grid_color):
        """Draw calibration grid for projector alignment"""
        glColor3f(*grid_color)
        glLineWidth(1.0)
        
        # Grid extends beyond table for alignment
        margin = 0.3
        start_x = -self.table.W/2 - margin
        end_x = self.table.W/2 + margin
        start_z = -self.table.L/2 - margin
        end_z = self.table.L/2 + margin
        
        glBegin(GL_LINES)
        
        # Vertical lines
        x = start_x
        while x <= end_x:
            glVertex3f(x, 0.003, start_z)
            glVertex3f(x, 0.003, end_z)
            x += self.grid_spacing
        
        # Horizontal lines
        z = start_z
        while z <= end_z:
            glVertex3f(start_x, 0.003, z)
            glVertex3f(end_x, 0.003, z)
            z += self.grid_spacing
        
        glEnd()
        
        # Draw center cross for alignment
        glLineWidth(3.0)
        glColor3f(1.0, 0.0, 0.0)  # Red center marker
        cross_size = 0.05
        
        glBegin(GL_LINES)
        glVertex3f(-cross_size, 0.004, 0)
        glVertex3f(cross_size, 0.004, 0)
        glVertex3f(0, 0.004, -cross_size)
        glVertex3f(0, 0.004, cross_size)
        glEnd()
        
        glLineWidth(1.0)
    
    def _update_trails(self, positions):
        """Update ball trails for smooth animation"""
        for i, pos in enumerate(positions):
            if i < len(self.physics._on_table) and self.physics._on_table[i]:
                if i not in self.ball_trails:
                    self.ball_trails[i] = []
                
                # Store position with table-relative Y
                trail_pos = (pos[0], pos[1] - self.table.H, pos[2])
                self.ball_trails[i].append(trail_pos)
                
                # Limit trail length
                if len(self.ball_trails[i]) > self.trail_length:
                    self.ball_trails[i].pop(0)
    
    def _draw_trails(self, trail_color):
        """Draw smooth ball trails"""
        glColor3f(*trail_color)
        glLineWidth(2.0)
        
        for ball_id, trail in self.ball_trails.items():
            if len(trail) < 2:
                continue
            
            glBegin(GL_LINE_STRIP)
            for i, (x, y, z) in enumerate(trail):
                # Fade trail over time
                alpha = (i + 1) / len(trail)
                glColor4f(trail_color[0], trail_color[1], trail_color[2], alpha * 0.8)
                glVertex3f(x, y + 0.005, z)  # Slightly above table
            glEnd()
        
        glLineWidth(1.0)
    
    def _draw_info_overlay(self):
        """Draw information overlay (for development mode)"""
        # This would require text rendering - simplified for now
        pass
    
    def _draw_help_overlay(self):
        """Draw help overlay"""
        # This would require text rendering - simplified for now
        pass


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Professional Pool Physics Projector Visualizer')
    parser.add_argument('--projector', action='store_true', help='Enable projector mode (top-down orthographic)')
    parser.add_argument('--fullscreen', action='store_true', help='Run in fullscreen mode')
    parser.add_argument('--width', type=int, default=1920, help='Window width')
    parser.add_argument('--height', type=int, default=1080, help='Window height')
    parser.add_argument('--rack', choices=['8-ball', '9-ball', '10-ball'], default='9-ball', help='Rack type')
    parser.add_argument('--speed', type=float, default=12.0, help='Break shot speed')
    parser.add_argument('--demo', action='store_true', help='Run demo break shot')
    
    args = parser.parse_args()
    
    print("Professional Pool Physics Projector Visualizer")
    print("=" * 50)
    
    # Initialize physics
    physics = PoolPhysics()
    table = PoolTable()
    
    # Set up rack
    positions = table.calc_racked_positions(args.rack, spacing_mode='random', seed=42)
    
    if args.rack == '9-ball':
        balls_on_table = list(range(10))
    elif args.rack == '10-ball':
        balls_on_table = list(range(11))
    else:  # 8-ball
        balls_on_table = list(range(16))
    
    physics.reset(ball_positions=positions, balls_on_table=balls_on_table)
    
    # Simulate break shot
    if args.demo:
        print(f"Simulating {args.rack} break at {args.speed} m/s...")
        events = physics.strike_ball(0.0, 0, 
                                   positions[0],
                                   positions[0] + np.array([0, 0, physics.ball_radius]),
                                   np.array([0.02, 0.0, -args.speed]), 0.54)
        
        print(f"Generated {len(events)} physics events")
    else:
        events = []
    
    # Create professional visualizer
    visualizer = ProjectorPoolVisualizer(
        physics, 
        width=args.width, 
        height=args.height,
        fullscreen=args.fullscreen,
        projector_mode=args.projector
    )
    
    print("\n🎱 Professional Pool Physics Visualizer 🎱")
    print("=" * 50)
    print("CONTROLS:")
    print("  SPACE     - Pause/Resume animation")
    print("  R         - Restart animation")
    print("  H         - Toggle help display")
    print("  T         - Toggle ball trails")
    print("  V         - Toggle velocity arrows")
    print("  G         - Toggle calibration grid")
    print("  C         - Toggle high contrast mode")
    print("  +/-       - Increase/decrease speed")
    print("  ESC       - Exit")
    print("=" * 50)
    
    if args.demo:
        # Run animation
        visualizer.animate_shot(events, speed=0.8)
    else:
        # Just show calibration mode
        visualizer.show_calibration_grid = True
        print("CALIBRATION MODE - Press G to toggle grid")
        visualizer.animate_shot([])  # Empty events for calibration


if __name__ == "__main__":
    main()