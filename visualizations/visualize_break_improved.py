#!/usr/bin/env python3
"""
Visualize pool break shots for different rack types.

This module provides functionality to simulate and visualize pool break shots
with before/after comparison views. It supports different rack types (8-ball,
9-ball, 10-ball) and various break speeds.

Typical usage example:
    python visualize_break_improved.py 9-ball 12.0
    python visualize_break_improved.py all 15.0
"""

import os
import sys
from typing import Tuple, List, Optional, Any

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pool_physics.table import PoolTable
from pool_physics import PoolPhysics
from pool_physics.events import BallSlidingEvent
from utils.table_renderer import draw_pool_table, draw_pocketed_balls


def simulate_break(
    rack_type: str = '8-ball', 
    break_speed: float = 12.0,
    spacing_mode: str = 'random',
    seed: int = None
) -> Tuple[PoolPhysics, NDArray[np.floating], NDArray[np.floating], List[Any], List[int]]:
    """
    Simulate a pool break shot for the given rack type.
    
    This function sets up a pool physics simulation, positions the balls in the
    specified rack configuration, and simulates a break shot by striking the
    cue ball toward the apex ball.
    
    Args:
        rack_type: Type of rack ('8-ball', '9-ball', or '10-ball').
        break_speed: Speed of the break shot in m/s.
        spacing_mode: Ball spacing mode ('random', 'fixed', or 'uniform').
        seed: Random seed for reproducible spacing (only used with 'random' mode).
        
    Returns:
        Tuple containing:
        - physics: PoolPhysics simulation object
        - initial_positions: Initial ball positions array
        - final_positions: Final ball positions after simulation
        - events: List of collision/motion events during simulation
        - balls_on_table: List of ball IDs that should be on table
        
    Raises:
        ValueError: If rack_type is not supported.
    """
    print(f"\nSimulating {rack_type} break at {break_speed} m/s...")
    
    # Validate rack type
    valid_rack_types = {'8-ball', '9-ball', '10-ball'}
    if rack_type not in valid_rack_types:
        raise ValueError(f"Rack type must be one of {valid_rack_types}, got '{rack_type}'")
    
    # Initialize physics simulation
    physics = PoolPhysics()
    table = PoolTable()
    
    # Get rack positions for specified rack type
    initial_positions = table.calc_racked_positions(rack_type=rack_type, spacing_mode=spacing_mode, seed=seed)
    
    # Determine which balls are on table based on rack type
    balls_on_table = _get_balls_for_rack_type(rack_type)
    
    # Reset physics with appropriate balls
    physics.reset(balls_on_table=balls_on_table, ball_positions=initial_positions)
    
    # Set up break shot parameters
    cue_ball_pos = initial_positions[0].copy()  # Cue ball at head spot
    target_pos = initial_positions[1].copy()    # Target apex ball at foot spot
    
    # Calculate direction vector from cue ball to target
    direction = target_pos - cue_ball_pos
    direction[1] = 0.0  # Keep shot horizontal (y=0)
    direction_norm = np.linalg.norm(direction)
    
    if direction_norm == 0:
        raise ValueError("Invalid table setup: cue ball and target ball at same position")
        
    direction_unit = direction / direction_norm
    
    # Set up cue strike parameters
    # Contact point slightly below center for slight draw
    r_c = cue_ball_pos.copy()
    r_c[1] -= physics.ball_radius * 0.1  # Hit 10% below center
    
    # Impact velocity (cue tip velocity)
    V = direction_unit * break_speed
    
    # Impact mass (effective mass of cue stick)
    M = 0.54  # Standard cue stick effective mass in kg
    
    print(f"Break direction: {direction_unit}")
    print(f"Contact point: {r_c}")
    print(f"Impact velocity: {V}")
    print(f"Impact mass: {M}")
    
    # Run simulation using realistic cue strike
    print("Running break simulation with cue strike...")
    events = physics.strike_ball(0.0, 0, cue_ball_pos, r_c, V, M)
    
    print(f"Generated {len(events)} events")
    
    # Get final positions after simulation settles (10 seconds should be enough)
    final_time = 10.0
    final_positions = physics.eval_positions(final_time)
    
    return physics, initial_positions, final_positions, events, balls_on_table


def _get_balls_for_rack_type(rack_type: str) -> List[int]:
    """Get list of ball IDs for specified rack type."""
    if rack_type == '9-ball':
        return list(range(10))  # Balls 0-9
    elif rack_type == '10-ball':
        return list(range(11))  # Balls 0-10
    else:  # 8-ball
        return list(range(16))  # Balls 0-15


def visualize_break_comparison(
    rack_type: str = '8-ball', 
    break_speed: float = 12.0,
    spacing_mode: str = 'random',
    seed: int = None
) -> Tuple[PoolPhysics, List[Any]]:
    """
    Create before/after visualization of break shot.
    
    This function generates a side-by-side comparison showing the table
    before and after the break shot, including proper display of pocketed balls.
    
    Args:
        rack_type: Type of rack ('8-ball', '9-ball', or '10-ball').
        break_speed: Initial speed of cue ball in m/s.
        spacing_mode: Ball spacing mode ('random', 'fixed', or 'tight').
        seed: Random seed for reproducible spacing (only used with 'random' mode).
        
    Returns:
        Tuple containing:
        - physics: PoolPhysics simulation object
        - events: List of events that occurred during simulation
    """
    # Run simulation
    physics, initial_pos, final_pos, events, balls_on_table = simulate_break(
        rack_type, break_speed, spacing_mode, seed
    )
    table = PoolTable()
    
    # Create figure with before/after subplots
    plt.close('all')  # Close any existing figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.set_dpi(100)
    
    # Table dimensions and styling
    ball_radius = table.ball_radius
    colors = plt.cm.Set3(np.linspace(0, 1, 16))
    
    # Draw both panels
    for ax, positions, title in [
        (ax1, initial_pos, 'Before Break'), 
        (ax2, final_pos, 'After Break')
    ]:
        # Draw table with pocketed ball area
        style_info = draw_pool_table(
            ax, table, style='standard', 
            show_pockets=True, show_spots=True, 
            show_grid=True, show_pocketed_area=True
        )
        
        # Draw balls currently on table
        _draw_on_table_balls(ax, positions, balls_on_table, ball_radius, colors, 
                           style_info, table, rack_type)
        
        # Draw pocketed balls below table
        draw_pocketed_balls(ax, positions, balls_on_table, ball_radius, colors, 
                           style_info, table, rack_type)
        
        # Set axis labels and title
        ax.set_xlabel('X Position (meters)')
        ax.set_ylabel('Z Position (meters)')
        ax.set_title(f'{title}\\n{rack_type.title()} at {break_speed} m/s')
    
    plt.tight_layout()
    
    # Save visualization
    os.makedirs('images', exist_ok=True)
    filename = f'images/{rack_type}_break_comparison.png'
    plt.savefig(filename, dpi=100, bbox_inches='tight', facecolor='white')
    print(f"Break comparison saved as '{filename}'")
    
    # Print statistics
    _print_break_statistics(rack_type, break_speed, events, final_pos, 
                          balls_on_table, table, ball_radius)
    
    plt.show()
    
    return physics, events


def _draw_on_table_balls(
    ax: matplotlib.axes.Axes,
    positions: NDArray[np.floating],
    balls_on_table: List[int], 
    ball_radius: float,
    colors: NDArray[np.floating],
    style_info: dict,
    table: Any,  # PoolTable object
    rack_type: str
) -> None:
    """Draw balls that are currently on the table surface."""
    for ball_id in balls_on_table:
        if ball_id < len(positions):
            x, y, z = positions[ball_id]
            
            # Check if ball is off table (pocketed or escaped through boundaries)
            if _is_ball_off_table(x, y, z, table, ball_radius):
                continue
                
            # Skip balls at origin for 9-ball and 10-ball (unused balls)
            if _is_unused_ball_at_origin(ball_id, x, z, rack_type):
                continue
            
            # Draw the ball
            circle = plt.Circle((x, z), ball_radius, 
                              color=colors[ball_id], alpha=style_info['ball_alpha'],
                              edgecolor=style_info['ball_edge_color'], 
                              linewidth=style_info['ball_edge_width'])
            ax.add_patch(circle)
            
            # Add ball number
            ax.text(x, z, str(ball_id), ha='center', va='center', 
                   fontsize=style_info['font_size'], fontweight='bold')


def _is_ball_off_table(x: float, y: float, z: float, table: Any, ball_radius: float) -> bool:
    """Check if ball is off the table (pocketed or escaped through boundaries)."""
    below_table = y < table.H - ball_radius
    outside_x = abs(x) > table.W/2 + ball_radius
    outside_z = abs(z) > table.L/2 + ball_radius
    
    return below_table or outside_x or outside_z


def _is_unused_ball_at_origin(ball_id: int, x: float, z: float, rack_type: str) -> bool:
    """Check if this is an unused ball at origin for 9-ball/10-ball games."""
    return (rack_type in ['9-ball', '10-ball'] and 
            ball_id > 0 and 
            abs(x) < 1e-6 and abs(z) < 1e-6)


def _print_break_statistics(
    rack_type: str,
    break_speed: float, 
    events: List[Any],
    final_pos: NDArray[np.floating],
    balls_on_table: List[int],
    table: Any,  # PoolTable object
    ball_radius: float
) -> None:
    """Print detailed break shot statistics."""
    print(f"\\nBreak Statistics:")
    print(f"Rack type: {rack_type}")
    print(f"Break speed: {break_speed} m/s")
    print(f"Total events: {len(events)}")
    
    # Count event types
    from pool_physics.events import BallCollisionEvent, RailCollisionEvent
    ball_collisions = len([e for e in events if isinstance(e, BallCollisionEvent)])
    rail_collisions = len([e for e in events if isinstance(e, RailCollisionEvent)])
    
    # Also check for SegmentCollisionEvent (actual rail collisions)
    segment_collisions = len([e for e in events if type(e).__name__ == 'SegmentCollisionEvent'])
    if segment_collisions > 0 and rail_collisions == 0:
        rail_collisions = segment_collisions
    
    print(f"Ball-to-ball collisions: {ball_collisions}")
    print(f"Rail collisions: {rail_collisions}")
    
    # Count balls remaining on table
    balls_on_table_final = 0
    for ball_id in balls_on_table:
        if ball_id < len(final_pos):
            x, y, z = final_pos[ball_id]
            
            # Skip unused balls at origin
            if _is_unused_ball_at_origin(ball_id, x, z, rack_type):
                continue
                
            # Check if ball is still on table
            if not _is_ball_off_table(x, y, z, table, ball_radius):
                balls_on_table_final += 1
    
    # Calculate initial count properly (excluding unused balls)
    balls_initially_active = _get_initial_active_ball_count(rack_type)
    balls_pocketed = balls_initially_active - balls_on_table_final
    
    print(f"Balls pocketed: {balls_pocketed}")
    print(f"Balls remaining: {balls_on_table_final}")


def _get_initial_active_ball_count(rack_type: str) -> int:
    """Get the number of balls initially active for the given rack type."""
    if rack_type == '9-ball':
        return 10  # Balls 0-9
    elif rack_type == '10-ball':
        return 11  # Balls 0-10
    else:  # 8-ball
        return 16  # Balls 0-15


def visualize_all_breaks(break_speed: float = 12.0) -> None:
    """Visualize break shots for all rack types."""
    for rack_type in ['8-ball', '9-ball', '10-ball']:
        print(f"\\n{'='*60}")
        print(f"Generating {rack_type} break visualization...")
        print(f"{'='*60}")
        visualize_break_comparison(rack_type, break_speed)


def main() -> None:
    """Main function for command-line usage."""
    if len(sys.argv) > 1:
        if sys.argv[1] == 'all':
            speed = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
            visualize_all_breaks(speed)
        else:
            rack_type = sys.argv[1]
            speed = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
            visualize_break_comparison(rack_type, speed)
    else:
        # Default behavior
        visualize_break_comparison()


if __name__ == "__main__":
    main()