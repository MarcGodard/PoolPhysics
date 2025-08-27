import os.path
import logging
_logger = logging.getLogger(__name__)
import numpy as np


INCH2METER = 0.0254
SQRT2 = np.sqrt(2)
DEG2RAD = np.pi/180


class PoolTable(object):
    """
    Pool table physics model with WPA-compliant dimensions and pocket geometry.
    
    Represents a standard 9-foot pool table with accurate rail geometry, pocket
    positions, and collision boundaries for realistic ball physics simulation.
    """
    
    def __init__(self,
                 # Table dimensions (WPA standard 9-foot table: 100" x 50")
                 L=100*INCH2METER,           # Table length (100 inches = 2.54m)
                 H=29.25*INCH2METER,         # Table height from floor (29.25 inches)
                 W=None,                     # Table width (defaults to L/2 = 50 inches)
                 
                 # Rail geometry parameters
                 ell_1=0.5*INCH2METER,       # Inner rail width (0.5")
                 ell_2=1.5*INCH2METER,       # Outer rail width (1.5")
                 h=1.575*INCH2METER,         # Rail height above playing surface
                 h_Q=1.625*INCH2METER,       # Rail height at contact point
                 r_P=0.1*INCH2METER,         # Rail corner radius
                 delta_QP=0.25*INCH2METER,   # Rail contact point offset
                 delta_PT=0.25*INCH2METER,   # Rail transition offset
                 a=1.75*INCH2METER,          # Rail angle parameter
                 A=60*DEG2RAD,               # Rail angle A (60 degrees)
                 C=60*DEG2RAD,               # Rail angle C (60 degrees)
                 
                 # Corner pocket geometry (WPA specifications)
                 M_cp=7.5*INCH2METER,        # Corner pocket mouth width (7.5")
                 T_cp=4*INCH2METER,          # Corner pocket throat width (4")
                 S_cp=1.75*INCH2METER,       # Corner pocket shelf depth
                 D_cp=2.5*INCH2METER,        # Corner pocket drop depth
                 r_cpc=2.625*INCH2METER,     # Corner pocket center radius
                 r_cpd=0.1875*INCH2METER,    # Corner pocket drop radius
                 
                 # Side pocket geometry (WPA specifications)
                 M_sp=6.5*INCH2METER,        # Side pocket mouth width (6.5")
                 T_sp=4.25*INCH2METER,       # Side pocket throat width (4.25")
                 S_sp=0,                     # Side pocket shelf depth (flush)
                 D_sp=1.25*INCH2METER,       # Side pocket drop depth
                 r_spc=2*INCH2METER,         # Side pocket center radius
                 r_spd=0.1875*INCH2METER,    # Side pocket drop radius
                 
                 # Ball and rail parameters
                 width_rail=None,            # Total rail width (auto-calculated)
                 ball_radius=1.125*INCH2METER,  # Standard ball radius (2.25" diameter)
                 num_balls=16,               # Total number of balls (cue + 15 object)
                 **kwargs):
        # Store table dimensions
        self.L = L  # Table length
        self.H = H  # Table height from floor
        if W is None:
            W = 0.5 * L  # Standard 2:1 aspect ratio (100" x 50")
        self.W = W  # Table width
        
        # Store rail geometry parameters
        self.ell_1 = ell_1  # Inner rail width
        self.ell_2 = ell_2  # Outer rail width
        self.w = ell_1 + ell_2  # Total rail width
        self.h = h  # Rail height
        
        # Store corner pocket parameters
        self.M_cp = M_cp  # Corner pocket mouth width
        self.T_cp = T_cp  # Corner pocket throat width
        self.S_cp = S_cp  # Corner pocket shelf depth
        self.D_cp = D_cp  # Corner pocket drop depth
        self.r_cpc = r_cpc  # Corner pocket center radius
        self.r_cpd = r_cpd  # Corner pocket drop radius
        
        # Store side pocket parameters
        self.M_sp = M_sp  # Side pocket mouth width
        self.T_sp = T_sp  # Side pocket throat width
        self.S_sp = S_sp  # Side pocket shelf depth
        self.D_sp = D_sp  # Side pocket drop depth
        self.r_spc = r_spc  # Side pocket center radius
        self.r_spd = r_spd  # Side pocket drop radius
        
        # Calculate total rail width if not specified
        if width_rail is None:
            width_rail = 1.5 * self.w  # Default: 1.5x the rail component width
        self.width_rail = width_rail
        
        # Store ball parameters
        self.ball_radius = ball_radius
        self.ball_diameter = 2 * ball_radius
        self.num_balls = num_balls
        self._almost_ball_radius = 0.999 * ball_radius  # Slightly smaller for collision detection
        # Calculate rail corner geometry (24 corner points defining table boundary)
        # This creates the complex rail shape around pockets for accurate collision detection
        corners = np.empty((24, 2))
        
        # Helper variables for geometry calculations
        w = 0.5 * W  # Half table width
        l = 0.5 * L  # Half table length  
        b = self.w   # Rail width
        
        # Pocket dimension shortcuts
        T_s, M_s = self.T_sp, self.M_sp  # Side pocket throat and mouth
        T_c, M_c = self.T_cp, self.M_cp  # Corner pocket throat and mouth
        
        # Define 24 corner points clockwise starting from bottom-left
        # Bottom edge (left to right): corners 0-3
        corners[0] = -(w + b) + T_c/SQRT2, -(l + b)        # Bottom-left rail outer corner
        corners[1] = -w + M_c/SQRT2, -l                    # Bottom-left pocket edge
        corners[2] = w - M_c/SQRT2, -l                     # Bottom-right pocket edge  
        corners[3] = w + b - T_c/SQRT2, -(l + b)          # Bottom-right rail outer corner
        
        # Right edge (bottom to top): corners 4-7
        corners[4] = w + b, -(l + b) + T_c/SQRT2          # Bottom-right rail corner
        corners[5] = w, -l + M_c/SQRT2                     # Bottom-right pocket corner
        corners[6] = w, -M_s/2                             # Right side pocket bottom edge
        corners[7] = w + b, -T_s/2                         # Right rail at side pocket bottom
        
        # Mirror bottom half to create top half (corners 8-15)
        corners[8] = corners[7, 0], -corners[7, 1]         # Right rail at side pocket top
        corners[9] = corners[6, 0], -corners[6, 1]         # Right side pocket top edge
        corners[10] = corners[5, 0], -corners[5, 1]        # Top-right pocket corner
        corners[11] = corners[4, 0], -corners[4, 1]        # Top-right rail corner
        corners[12] = corners[3, 0], -corners[3, 1]        # Top-right rail outer corner
        corners[13] = corners[2, 0], -corners[2, 1]        # Top-right pocket edge
        corners[14] = corners[1, 0], -corners[1, 1]        # Top-left pocket edge
        corners[15] = corners[0, 0], -corners[0, 1]        # Top-left rail outer corner
        
        # Mirror right half to create left half (corners 16-23)
        corners[16] = -corners[11, 0], corners[11, 1]      # Top-left rail corner
        corners[17] = -corners[10, 0], corners[10, 1]      # Top-left pocket corner
        corners[18] = -corners[9, 0], corners[9, 1]        # Left side pocket top edge
        corners[19] = -corners[8, 0], corners[8, 1]        # Left rail at side pocket top
        corners[20] = -corners[7, 0], corners[7, 1]        # Left rail at side pocket bottom
        corners[21] = -corners[6, 0], corners[6, 1]        # Left side pocket bottom edge
        corners[22] = -corners[5, 0], corners[5, 1]        # Bottom-left pocket corner
        corners[23] = -corners[4, 0], corners[4, 1]        # Bottom-left rail corner
        
        self._corners = corners
        # Initialize pocket positions array (6 pockets: 4 corners + 2 sides)
        self.pocket_positions = np.zeros((6, 3), dtype=np.float64)
        self.pocket_positions[:, 1] = H  # All pockets at table height
        
        # Define WPA standard pocket sizes for collision detection
        # Corner pockets: 4.5-4.625" diameter (2-2.06 ball diameters)
        # Side pockets: 5.0-5.25" diameter (2.22-2.33 ball diameters)
        R_c = 2 * self.ball_radius      # Corner pocket radius: 4.5" diameter (2.0 ball diameters)
        R_s = 2.33 * self.ball_radius   # Side pocket radius: 5.24" diameter (2.33 ball diameters)
        self.R_c, self.R_s = R_c, R_s
        
        # Position pockets according to standard pool table layout
        # Pocket numbering: 0=bottom-left, 1=bottom-right, 2=right-side, 
        #                   3=top-right, 4=top-left, 5=left-side
        
        # Corner pockets at table corners
        self.pocket_positions[0, ::2] = [-self.W/2, -self.L/2]  # Bottom-left corner
        self.pocket_positions[1, ::2] = [ self.W/2, -self.L/2]  # Bottom-right corner  
        self.pocket_positions[3, ::2] = [ self.W/2,  self.L/2]  # Top-right corner
        self.pocket_positions[4, ::2] = [-self.W/2,  self.L/2]  # Top-left corner
        
        # Side pockets at midpoint of long rails
        self.pocket_positions[2, ::2] = [ self.W/2,  0.0]       # Right side pocket
        self.pocket_positions[5, ::2] = [-self.W/2,  0.0]       # Left side pocket

    def corner_to_pocket(self, i_c):
        return (i_c + 2) % 24 // 4

    def pocket_to_corner(self, i_p):
        return i_p * 4 - 2

    def is_position_in_bounds(self, r):
        """ r: position vector; R: ball radius """
        R = self._almost_ball_radius
        return  -0.5*self.W + R <= r[0] <= 0.5*self.W - R \
            and -0.5*self.L + R <= r[2] <= 0.5*self.L - R

    def is_position_near_pocket(self, r):
        """ r: position vector; R: ball radius """
        if r[0] < -0.5*self.W + self.M_cp/np.sqrt(2):
            if r[2] < -0.5*self.L + self.M_cp/np.sqrt(2):
                _logger.info('corner pocket 0')
                return 0
            elif r[2] > 0.5*self.L - self.M_cp/np.sqrt(2):
                _logger.info('corner pocket 1')
                return 1
        elif r[0] > 0.5*self.W - self.M_cp/np.sqrt(2):
            if r[2] < -0.5*self.L + self.M_cp/np.sqrt(2):
                _logger.info('corner pocket 2')
                return 2
            elif r[2] > 0.5*self.L - self.M_cp/np.sqrt(2):
                _logger.info('corner pocket 3')
                return 3

    def calc_racked_positions(self, rack_type='8-ball', d=None, out=None):
        """
        Calculate racked positions for different pool games
        
        Args:
            rack_type: '8-ball', '9-ball', or '10-ball'
            d: spacing between balls (default: random 0.01-0.2mm gap)
            out: output array (default: create new array)
        """
        if out is None:
            out = np.empty((self.num_balls, 3), dtype=np.float64)
        ball_radius = self.ball_radius
        if d is None:
            # Random gap between 0.01mm and 0.2mm (converted to meters)
            d = np.random.uniform(0.01e-3, 0.2e-3)
        length = self.L
        ball_diameter = 2*ball_radius
        
        # Set y-position (height) for all balls
        out[:,1] = self.H + ball_radius
        
        # Calculate spacing between balls
        spacing = ball_diameter + d
        row_spacing = spacing * np.sqrt(3) / 2
        
        # Foot spot position (standard: 1/4 table length from foot rail)
        foot_spot_z = -0.25 * length
        
        # Initialize all positions to zero
        out[1:, 0] = 0.0
        out[1:, 2] = 0.0
        
        if rack_type == '8-ball':
            # 15-ball triangle rack
            rack_positions = [
                # Row 1 (apex) - ball 1 at foot spot
                (1,),
                # Row 2 
                (2, 6),
                # Row 3 - ball 8 in center
                (10, 8, 3),
                # Row 4
                (13, 11, 7, 4),
                # Row 5 (back)
                (15, 14, 12, 9, 5)
            ]
            
        elif rack_type == '9-ball':
            # 9-ball diamond rack (proper diamond formation)
            rack_positions = [
                # Row 1 (apex) - ball 1 at foot spot
                (1,),
                # Row 2
                (2, 3),
                # Row 3 - ball 9 in center (widest part of diamond)
                (4, 9, 5),
                # Row 4
                (6, 7),
                # Row 5 (back point of diamond)
                (8,)
            ]
            
        elif rack_type == '10-ball':
            # 10-ball triangle rack
            rack_positions = [
                # Row 1 (apex) - ball 1 at foot spot
                (1,),
                # Row 2
                (2, 3),
                # Row 3 - ball 10 in center
                (4, 10, 5),
                # Row 4
                (6, 7, 8, 9)
            ]
            
        else:
            raise ValueError(f"Unknown rack_type: {rack_type}. Must be '8-ball', '9-ball', or '10-ball'")
        
        # Calculate geometric positions
        for row_num, balls_in_row in enumerate(rack_positions):
            z_pos = foot_spot_z - row_num * row_spacing
            num_balls = len(balls_in_row)
            
            # Center the row
            if num_balls == 1:
                x_positions = [0.0]
            else:
                x_start = -(num_balls - 1) * spacing / 2
                x_positions = [x_start + i * spacing for i in range(num_balls)]
            
            # Assign positions to specific ball numbers
            for i, ball_num in enumerate(balls_in_row):
                out[ball_num, 0] = x_positions[i]
                out[ball_num, 2] = z_pos
        
        # cue ball at head spot:
        out[0,0] = 0.0
        out[0,2] = 0.25 * length
        return out
