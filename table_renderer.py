#!/usr/bin/env python3
"""
Unified table rendering functions for consistent visualization across all scripts
"""
import matplotlib.pyplot as plt
import numpy as np


def draw_pool_table(ax, table, style='standard', show_pockets=True, show_spots=True, show_grid=True):
    """
    Draw a standardized pool table with consistent styling (based on rack visualization)
    
    Args:
        ax: matplotlib axis to draw on
        table: PoolTable object
        style: 'standard', 'compact', or 'minimal'
        show_pockets: whether to draw pockets
        show_spots: whether to draw foot/head spots
        show_grid: whether to show reference grid
    
    Returns:
        dict with styling info for ball rendering consistency
    """
    table_width = table.W
    table_length = table.L
    
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


def draw_balls(ax, positions, balls_to_show, ball_radius, colors, style_info, rack_type=None):
    """
    Draw balls with consistent styling
    
    Args:
        ax: matplotlib axis
        positions: ball positions array
        balls_to_show: list of ball IDs to display
        ball_radius: radius of balls
        colors: color array for balls
        style_info: styling dict from draw_pool_table
        rack_type: optional rack type for filtering unused balls
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


def setup_axis_labels(ax, title, style='standard', xlabel='X Position (meters)', ylabel='Z Position (meters)'):
    """
    Set up axis labels and titles consistently
    
    Args:
        ax: matplotlib axis
        title: plot title
        style: visualization style
        xlabel, ylabel: axis labels
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