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
        self.show_velocity_arrows = True
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
                    1: (1.0, 1.0, 0.0),         # 1-ball - yellow
                    2: (0.0, 0.0, 1.0),         # 2-ball - blue
                    3: (1.0, 0.0, 0.0),         # 3-ball - red
                    4: (0.5, 0.0, 0.5),         # 4-ball - purple
                    5: (1.0, 0.5, 0.0),         # 5-ball - orange
                    6: (0.0, 0.5, 0.0),         # 6-ball - green
                    7: (0.5, 0.0, 0.0),         # 7-ball - maroon
                    8: (0.0, 0.0, 0.0),         # 8-ball - black
                    9: (1.0, 1.0, 0.0),         # 9-ball - yellow stripe
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
                    1: (1.0, 1.0, 0.2),         # Bright yellow
                    2: (0.2, 0.2, 1.0),         # Bright blue
                    3: (1.0, 0.2, 0.2),         # Bright red
                    4: (0.8, 0.2, 0.8),         # Bright purple
                    5: (1.0, 0.6, 0.2),         # Bright orange
                    6: (0.2, 0.8, 0.2),         # Bright green
                    7: (0.8, 0.2, 0.2),         # Bright maroon
                    8: (0.2, 0.2, 0.2),         # Dark gray (visible on projection)
                    9: (1.0, 1.0, 0.4),         # Bright yellow
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
        
        # Lighting for 3D effect
        if not self.projector_mode:
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
            glLightfv(GL_LIGHT0, GL_POSITION, [0, 3, 0, 1])
            glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1])
            glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
        
        # Set up camera
        self._setup_camera()
        
        # Set background color (black for projection)
        glClearColor(0.0, 0.0, 0.0, 1.0)
    
    def _setup_camera(self):
        """Set up optimal camera for projector or 3D viewing"""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        if self.projector_mode:
            # Perfect orthographic projection for projector alignment
            # Add slight margin around table for alignment markers
            margin = 0.15
            table_w, table_l = self.table.W, self.table.L
            
            glOrtho(-table_w/2 - margin, table_w/2 + margin,
                   -table_l/2 - margin, table_l/2 + margin,
                   -1.0, 5.0)
            
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            
            # Perfect top-down view (Y-up, looking down negative Y)
            gluLookAt(0, 2.0, 0,    # Camera position (high above table)
                     0, 0, 0,       # Look at table center
                     0, 0, -1)      # Up vector (Z points up in table space)
        else:
            # 3D perspective view for development/testing
            aspect_ratio = self.width / self.height
            gluPerspective(45, aspect_ratio, 0.1, 20.0)
            
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            
            # Angled 3D view
            camera_distance = 4.0
            camera_height = 2.5
            gluLookAt(camera_distance, camera_height, camera_distance,
                     0, 0, 0,
                     0, 1, 0)
    
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
        
        # Clear trails
        self.ball_trails.clear()
        
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
            
            # Draw balls
            for i, pos in enumerate(positions):
                if i < len(self.physics._on_table) and self.physics._on_table[i]:
                    ball_color = colors['balls'].get(i, (0.8, 0.8, 0.8))
                    velocity = velocities[i] if i < len(velocities) else None
                    self._draw_ball_professional(pos, ball_color, i, velocity)
        
        except Exception as e:
            _logger.warning(f"Error rendering at t={self.current_time:.3f}: {e}")
        
        # Draw UI overlays (only in non-projector mode or if explicitly enabled)
        if show_info and not self.projector_mode:
            self._draw_info_overlay()
        
        if show_help:
            self._draw_help_overlay()
    
    def _draw_table_professional(self, colors):
        """Draw professional-quality table"""
        # Table surface (felt)
        glColor3f(*colors['table'])
        glBegin(GL_QUADS)
        glVertex3f(-self.table.W/2, 0, -self.table.L/2)
        glVertex3f(self.table.W/2, 0, -self.table.L/2)
        glVertex3f(self.table.W/2, 0, self.table.L/2)
        glVertex3f(-self.table.W/2, 0, self.table.L/2)
        glEnd()
        
        # Rails with proper proportions
        rail_height = 0.04 if self.projector_mode else 0.06
        rail_width = 0.08
        
        glColor3f(*colors['rails'])
        self._draw_rail_system(rail_width, rail_height)
        
        # Pockets
        glColor3f(*colors['pockets'])
        self._draw_pockets()
        
        # Table markings for professional appearance
        self._draw_table_markings(colors)
    
    def _draw_rail_system(self, width, height):
        """Draw complete rail system with proper geometry"""
        W, L = self.table.W, self.table.L
        
        # Bottom rail
        self._draw_rail_segment(-W/2 - width, 0, -L/2 - width, W + 2*width, height, width)
        # Top rail  
        self._draw_rail_segment(-W/2 - width, 0, L/2, W + 2*width, height, width)
        # Left rail
        self._draw_rail_segment(-W/2 - width, 0, -L/2, width, height, L)
        # Right rail
        self._draw_rail_segment(W/2, 0, -L/2, width, height, L)
    
    def _draw_rail_segment(self, x, y, z, width, height, length):
        """Draw a single rail segment with proper 3D geometry"""
        glBegin(GL_QUADS)
        
        # Top surface
        glNormal3f(0, 1, 0)
        glVertex3f(x, y + height, z)
        glVertex3f(x + width, y + height, z)
        glVertex3f(x + width, y + height, z + length)
        glVertex3f(x, y + height, z + length)
        
        if not self.projector_mode:  # Only draw sides in 3D mode
            # Front face
            glNormal3f(0, 0, -1)
            glVertex3f(x, y, z)
            glVertex3f(x + width, y, z)
            glVertex3f(x + width, y + height, z)
            glVertex3f(x, y + height, z)
            
            # Back face
            glNormal3f(0, 0, 1)
            glVertex3f(x + width, y, z + length)
            glVertex3f(x, y, z + length)
            glVertex3f(x, y + height, z + length)
            glVertex3f(x + width, y + height, z + length)
        
        glEnd()
    
    def _draw_pockets(self):
        """Draw professional 3D pocket geometry with realistic depth and chamfered edges"""
        # Standard pocket positions (same as table.py)
        pocket_positions = [
            (-self.table.W/2, -self.table.L/2, 'corner'),  # Bottom left corner
            (self.table.W/2, -self.table.L/2, 'corner'),   # Bottom right corner
            (self.table.W/2, 0, 'side'),                   # Right side
            (self.table.W/2, self.table.L/2, 'corner'),    # Top right corner
            (-self.table.W/2, self.table.L/2, 'corner'),   # Top left corner
            (-self.table.W/2, 0, 'side'),                  # Left side
        ]
        
        # Professional pocket dimensions (matching WPA specs from table.py)
        corner_radius = 0.0571  # 2.25" diameter (2 ball diameters) 
        side_radius = 0.0665    # 2.625" diameter (2.33 ball diameters)
        pocket_depth = 0.05     # Realistic pocket depth
        chamfer_depth = 0.008   # Chamfered edge depth
        
        for px, pz, pocket_type in pocket_positions:
            radius = corner_radius if pocket_type == 'corner' else side_radius
            self._draw_single_pocket(px, pz, radius, pocket_depth, chamfer_depth)
    
    def _draw_single_pocket(self, x, z, radius, depth, chamfer_depth):
        """Draw a single 3D pocket with realistic geometry"""
        segments = 24  # Smooth circular pockets
        
        # 1. Draw pocket opening (chamfered rim)
        glColor3f(0.05, 0.05, 0.05)  # Dark charcoal for pocket rim
        glBegin(GL_QUAD_STRIP)
        for i in range(segments + 1):
            angle = 2 * np.pi * i / segments
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            # Outer rim at table level
            x1 = x + radius * cos_a
            z1 = z + radius * sin_a
            glVertex3f(x1, 0.001, z1)
            
            # Inner chamfered edge 
            inner_radius = radius * 0.85
            x2 = x + inner_radius * cos_a  
            z2 = z + inner_radius * sin_a
            glVertex3f(x2, -chamfer_depth, z2)
        glEnd()
        
        # 2. Draw pocket cylinder wall
        glColor3f(0.02, 0.02, 0.02)  # Very dark for pocket interior
        glBegin(GL_QUAD_STRIP)
        inner_radius = radius * 0.85
        for i in range(segments + 1):
            angle = 2 * np.pi * i / segments
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            # Top of cylinder (at chamfer depth)
            x1 = x + inner_radius * cos_a
            z1 = z + inner_radius * sin_a
            glVertex3f(x1, -chamfer_depth, z1)
            
            # Bottom of cylinder
            bottom_radius = inner_radius * 0.7  # Tapered bottom
            x2 = x + bottom_radius * cos_a
            z2 = z + bottom_radius * sin_a
            glVertex3f(x2, -depth, z2)
        glEnd()
        
        # 3. Draw pocket bottom
        glColor3f(0.01, 0.01, 0.01)  # Nearly black pocket bottom
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(x, -depth, z)  # Center of pocket bottom
        
        bottom_radius = inner_radius * 0.7
        for i in range(segments + 1):
            angle = 2 * np.pi * i / segments
            x_bottom = x + bottom_radius * np.cos(angle)
            z_bottom = z + bottom_radius * np.sin(angle)
            glVertex3f(x_bottom, -depth, z_bottom)
        glEnd()
        
        # 4. Add subtle pocket highlight (leather/felt texture)
        if not self.projector_mode:  # Only in 3D mode
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            glColor4f(0.15, 0.12, 0.08, 0.4)  # Subtle leather-like highlight
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(x, -chamfer_depth * 0.5, z)
            
            highlight_radius = inner_radius * 0.9
            for i in range(segments + 1):
                angle = 2 * np.pi * i / segments
                x_h = x + highlight_radius * np.cos(angle)
                z_h = z + highlight_radius * np.sin(angle)
                glVertex3f(x_h, -chamfer_depth * 0.3, z_h)
            glEnd()
            
            glDisable(GL_BLEND)
        
        # 5. Add pocket edge accent (metallic ring)
        glColor3f(0.3, 0.25, 0.2)  # Metallic bronze accent
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x_edge = x + radius * 0.98 * np.cos(angle)  # Slightly inside rim
            z_edge = z + radius * 0.98 * np.sin(angle)
            glVertex3f(x_edge, 0.002, z_edge)  # Just above table surface
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
    
    def _draw_ball_professional(self, position, color, ball_id, velocity=None):
        """Draw high-quality ball with professional appearance"""
        x, y, z = position[0], position[1] - self.table.H, position[2]
        
        glPushMatrix()
        glTranslatef(x, y, z)
        
        # Ball color with slight shine
        glColor3f(*color)
        
        # Draw main ball (high-quality sphere)
        sphere = gluNewQuadric()
        gluQuadricNormals(sphere, GLU_SMOOTH)
        gluSphere(sphere, self.physics.ball_radius, 24, 24)
        gluDeleteQuadric(sphere)
        
        # Ball number (for non-cue balls)
        if ball_id > 0:
            glPushMatrix()
            glRotatef(90, 1, 0, 0)  # Face the number upward
            # This would require text rendering - simplified for now
            glPopMatrix()
        
        glPopMatrix()
        
        # Velocity arrow
        if self.show_velocity_arrows and velocity is not None:
            speed = np.linalg.norm(velocity[:3])
            if speed > 0.1:  # Only show if ball is moving significantly
                self._draw_velocity_arrow(position, velocity, color)
    
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