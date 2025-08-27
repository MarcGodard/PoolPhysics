#!/usr/bin/env python3
"""
Unified table rendering functions for consistent visualization across all scripts.

This module provides standardized functions for rendering pool tables and balls
with consistent styling across different visualization scripts.

Typical usage example:
    table = PoolTable()
    fig, ax = plt.subplots()
    style_info = draw_pool_table(ax, table)
    draw_balls(ax, positions, balls_to_show, ball_radius, colors, style_info)
"""
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.axes
import numpy as np
from numpy.typing import NDArray


def draw_pool_table(
    ax: matplotlib.axes.Axes, 
    table: Any,  # PoolTable object
    style: str = 'standard', 
    show_pockets: bool = True, 
    show_spots: bool = True, 
    show_grid: bool = True, 
    show_pocketed_area: bool = False
) -> Dict[str, Any]:
    """
    Draw a standardized pool table with consistent styling.
    
    This function draws a pool table with configurable styling options including
    pockets, spots, grid lines, and optional area for showing pocketed balls.
    
    Args:
        ax: Matplotlib axis to draw on.
        table: PoolTable object containing table dimensions and properties.
        style: Visual style - 'standard', 'compact', or 'minimal'.
        show_pockets: Whether to draw pocket circles.
        show_spots: Whether to draw foot/head spots.
        show_grid: Whether to show reference grid lines.
        show_pocketed_area: Whether to extend view to show pocketed balls area.
    
    Returns:
        Dictionary containing styling information for consistent ball rendering:
        - ball_alpha: Alpha transparency for balls
        - ball_edge_color: Color for ball edges
        - ball_edge_width: Width of ball edge lines
        - font_size: Font size for ball numbers
        - margin: Margin around table
        
    Raises:
        ValueError: If style parameter is not recognized.
    """
    table_width = table.W
    table_length = table.L
    
    # Validate style parameter
    valid_styles = {'standard', 'compact', 'minimal'}
    if style not in valid_styles:
        raise ValueError(f"Style must be one of {valid_styles}, got '{style}'")
    
    # Style configurations (based on best-looking rack visualization)
    if style == 'compact':
        # For timelapse/small panels
        boundary_width = 1.5
        pocket_alpha = 0.6
        pocket_edge_width = 0.5
        spot_size = 3
        spot_alpha = 0.5
        grid_alpha = 0.1
        reference_alpha = 0.2
        margin = 0.02
    elif style == 'minimal':
        # For clean presentations
        boundary_width = 2
        pocket_alpha = 0.7
        pocket_edge_width = 0.8
        spot_size = 6
        spot_alpha = 0.6
        grid_alpha = 0.2
        reference_alpha = 0.3
        margin = 0.05
    else:  # 'standard' - matches rack visualization exactly
        boundary_width = 2
        pocket_alpha = 0.8
        pocket_edge_width = 1
        spot_size = 8
        spot_alpha = 0.7
        grid_alpha = 0.3
        reference_alpha = 0.5
        margin = 0.1
    
    # Draw table boundaries
    ax.plot([-table_width/2, table_width/2, table_width/2, -table_width/2, -table_width/2],
            [-table_length/2, -table_length/2, table_length/2, table_length/2, -table_length/2],
            'k-', linewidth=boundary_width)
    
    # Draw pockets (matching rack visualization style exactly)
    if show_pockets:
        for i, pocket_pos in enumerate(table.pocket_positions):
            pocket_radius = table.R_c if i in [0, 1, 3, 4] else table.R_s
            pocket_color = 'darkred' if i in [0, 1, 3, 4] else 'darkblue'
            
            circle = plt.Circle((pocket_pos[0], pocket_pos[2]), pocket_radius, 
                              color=pocket_color, alpha=pocket_alpha, 
                              edgecolor='black', linewidth=pocket_edge_width)
            ax.add_patch(circle)
    
    # Draw reference lines
    if show_grid or reference_alpha > 0:
        ax.axhline(y=0, color='gray', linestyle='--', alpha=reference_alpha)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=reference_alpha)
    
    # Draw spots
    if show_spots:
        ax.plot(0, -table_length/4, 'ro', markersize=spot_size, alpha=spot_alpha)  # Foot spot
        ax.plot(0, table_length/4, 'bo', markersize=spot_size, alpha=spot_alpha)   # Head spot
    
    # Set consistent axis properties
    margin = 0.02 if style == 'compact' else 0.05 if style == 'minimal' else 0.1
    ax.set_xlim(-table_width/2 - margin, table_width/2 + margin)
    
    # Extend y-axis if showing pocketed ball area
    if show_pocketed_area:
        pocketed_area_height = 0.25  # Extra space for pocketed balls and labels
        ax.set_ylim(-table_length/2 - pocketed_area_height, table_length/2 + margin)
    else:
        ax.set_ylim(-table_length/2 - margin, table_length/2 + margin)
    
    ax.set_aspect('equal')
    
    if show_grid:
        ax.grid(True, alpha=grid_alpha)
    
    # Return styling info for consistent ball rendering (matching rack style)
    return {
        'ball_alpha': 0.7,  # Matches rack visualization
        'ball_edge_color': 'black',
        'ball_edge_width': 0.5 if style == 'compact' else 1.0,
        'font_size': 6 if style == 'compact' else 8,  # Matches rack fontsize
        'margin': margin
    }


def draw_balls(
    ax: matplotlib.axes.Axes, 
    positions: NDArray[np.floating], 
    balls_to_show: List[int], 
    ball_radius: float, 
    colors: NDArray[np.floating], 
    style_info: Dict[str, Any], 
    rack_type: Optional[str] = None
) -> None:
    """
    Draw balls with consistent styling.
    
    This function draws pool balls at their specified positions with consistent
    styling based on the provided style information.
    
    Args:
        ax: Matplotlib axis to draw on.
        positions: Array of ball positions with shape (n_balls, 3) for (x, y, z).
        balls_to_show: List of ball IDs to display.
        ball_radius: Radius of balls in meters.
        colors: Color array for balls with shape (n_balls, 3) or (n_balls, 4).
        style_info: Styling dictionary from draw_pool_table containing visual parameters.
        rack_type: Optional rack type ('8-ball', '9-ball', '10-ball') for filtering unused balls.
        
    Note:
        For 9-ball and 10-ball games, unused balls at the origin are automatically filtered out.
    """
    for ball_id in balls_to_show:
        if ball_id < len(positions):
            x, _, z = positions[ball_id]
            
            # Skip balls at origin for 9-ball and 10-ball (unused balls)
            if rack_type in ['9-ball', '10-ball'] and ball_id > 0 and abs(x) < 1e-6 and abs(z) < 1e-6:
                continue
            
            # Skip balls that are off table or pocketed (very low y position)  
            # Note: This check would need the full 3D position, commenting out for now
            # if y < table.H - ball_radius:
            #     continue
                
            circle = plt.Circle((x, z), ball_radius, 
                              color=colors[ball_id], alpha=style_info['ball_alpha'],
                              edgecolor=style_info['ball_edge_color'], 
                              linewidth=style_info['ball_edge_width'])
            ax.add_patch(circle)
            
            # Add ball number
            ax.text(x, z, str(ball_id), ha='center', va='center', 
                   fontsize=style_info['font_size'], fontweight='bold')


def draw_pocketed_balls(
    ax: matplotlib.axes.Axes, 
    all_positions: NDArray[np.floating], 
    balls_to_check: List[int], 
    ball_radius: float, 
    colors: NDArray[np.floating], 
    style_info: Dict[str, Any], 
    table: Any,  # PoolTable object
    rack_type: Optional[str] = None
) -> None:
    """
    Draw pocketed balls in a line below the table.
    
    This function identifies balls that are off the table (either pocketed or escaped
    through boundaries) and displays them in a horizontal line below the table with
    a "Pocketed" label.
    
    Args:
        ax: Matplotlib axis to draw on.
        all_positions: Array of all ball positions with shape (n_balls, 3) for (x, y, z).
        balls_to_check: List of ball IDs to check for pocketed status.
        ball_radius: Radius of balls in meters.
        colors: Color array for balls with shape (n_balls, 3) or (n_balls, 4).
        style_info: Styling dictionary from draw_pool_table containing visual parameters.
        table: PoolTable object for reference dimensions and height.
        rack_type: Optional rack type ('8-ball', '9-ball', '10-ball') for filtering unused balls.
        
    Note:
        Balls are considered "pocketed" if they are:
        - Below the table surface (y < table.H - ball_radius), OR
        - Outside the X boundaries (|x| > table.W/2 + ball_radius), OR  
        - Outside the Z boundaries (|z| > table.L/2 + ball_radius)
    """
    pocketed_balls = []
    
    for ball_id in balls_to_check:
        if ball_id < len(all_positions):
            x, y, z = all_positions[ball_id]
            
            # Skip balls at origin for 9-ball and 10-ball (unused balls)
            if rack_type in ['9-ball', '10-ball'] and ball_id > 0 and abs(x) < 1e-6 and abs(z) < 1e-6:
                continue
            
            # Check if ball is off table (pocketed or escaped)
            # Either below table surface OR outside table boundaries
            below_table = y < table.H - ball_radius
            outside_x = abs(x) > table.W/2 + ball_radius
            outside_z = abs(z) > table.L/2 + ball_radius
            
            if below_table or outside_x or outside_z:
                pocketed_balls.append(ball_id)
    
    # Draw pocketed balls in a line below the table
    if pocketed_balls:
        table_bottom = -table.L/2
        pocket_area_y = table_bottom - 0.15  # Position below table
        
        # Calculate spacing for pocketed balls
        total_width = len(pocketed_balls) * ball_radius * 2 + (len(pocketed_balls) - 1) * ball_radius * 0.5
        start_x = -total_width / 2
        
        for i, ball_id in enumerate(pocketed_balls):
            x_pos = start_x + i * (ball_radius * 2.5)  # 2.5 gives nice spacing
            
            circle = plt.Circle((x_pos, pocket_area_y), ball_radius, 
                              color=colors[ball_id], alpha=style_info['ball_alpha'],
                              edgecolor=style_info['ball_edge_color'], 
                              linewidth=style_info['ball_edge_width'])
            ax.add_patch(circle)
            
            # Add ball number
            ax.text(x_pos, pocket_area_y, str(ball_id), ha='center', va='center', 
                   fontsize=style_info['font_size'], fontweight='bold')
        
        # Add "Pocketed" label
        label_y = pocket_area_y - ball_radius * 2
        ax.text(0, label_y, 'Pocketed', ha='center', va='center', 
               fontsize=style_info['font_size'], fontweight='bold', color='gray')


def setup_axis_labels(
    ax: matplotlib.axes.Axes, 
    title: str, 
    style: str = 'standard', 
    xlabel: str = 'X Position (meters)', 
    ylabel: str = 'Z Position (meters)'
) -> None:
    """
    Set up axis labels and titles consistently.
    
    This function configures axis labels, titles, and tick marks based on the
    specified visualization style.
    
    Args:
        ax: Matplotlib axis to configure.
        title: Plot title to display.
        style: Visualization style ('standard', 'compact', 'minimal').
        xlabel: X-axis label.
        ylabel: Y-axis label (note: represents Z position in 3D space).
        
    Note:
        For 'compact' style, axis ticks are removed for cleaner timelapse panels.
    """
    if style == 'compact':
        # Remove axis labels for timelapse panels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9, fontweight='bold')
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        title_size = 12 if style == 'minimal' else 14
        ax.set_title(title, fontsize=title_size, fontweight='bold')