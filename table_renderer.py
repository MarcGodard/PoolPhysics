#!/usr/bin/env python3
"""
Unified table rendering functions for consistent visualization across all scripts
"""
import matplotlib.pyplot as plt
import numpy as np


def draw_pool_table(ax, table, style='standard', show_pockets=True, show_spots=True, show_grid=True, show_pocketed_area=False):
    """
    Draw a standardized pool table with consistent styling (based on rack visualization)
    
    Args:
        ax: matplotlib axis to draw on
        table: PoolTable object
        style: 'standard', 'compact', or 'minimal'
        show_pockets: whether to draw pockets
        show_spots: whether to draw foot/head spots
        show_grid: whether to show reference grid
        show_pocketed_area: whether to extend view to show pocketed balls area
    
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


def draw_pocketed_balls(ax, all_positions, balls_to_check, ball_radius, colors, style_info, table, rack_type=None):
    """
    Draw pocketed balls in a line below the table
    
    Args:
        ax: matplotlib axis
        all_positions: all ball positions array (3D with y component)
        balls_to_check: list of ball IDs to check for pocketed status
        ball_radius: radius of balls
        colors: color array for balls
        style_info: styling dict from draw_pool_table
        table: PoolTable object for reference height
        rack_type: optional rack type for filtering unused balls
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