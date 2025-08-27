# Pool Physics Development Guide

## Project Structure

The Pool Physics project has been organized into a clean, maintainable structure following Python best practices:

```
PoolPhysics/
├── pool_physics/                  # Core physics engine
│   ├── __init__.py               # Main PoolPhysics class
│   ├── events.py                 # Event system (collisions, motion)  
│   ├── table.py                  # Pool table definitions
│   ├── collisions.py             # Collision physics
│   └── poly_solvers.py           # Mathematical solvers
├── visualizations/               # Visualization scripts  
│   ├── visualize_break.py        # Break shot comparisons
│   ├── visualize_break_improved.py  # Improved version with type hints
│   ├── visualize_break_timelapse.py # Time-series break analysis
│   └── visualize_rack.py         # Rack position visualization
├── utils/                        # Utility modules
│   └── table_renderer.py         # Consistent table/ball rendering
├── tests/                        # Test files
│   ├── test_rack_arrangement.py  # Rack positioning tests
│   └── test_rail_collision.py    # Rail collision tests
├── debug_tools/                  # Debug and analysis tools
│   ├── analyze_segments.py       # Table segment analysis
│   ├── debug_ball_positions.py   # Ball position debugging
│   ├── debug_cue_ball_escape.py  # Cue ball escape analysis
│   └── debug_events.py           # Event system debugging
├── test/                         # Original test directory (pytest)
├── images/                       # Generated visualization outputs
└── scripts/                      # Build and deployment scripts
```

## Code Quality Standards

### Python 3 Standards
- **Type Hints**: All new code uses proper type annotations
- **Docstrings**: Google-style docstrings for all functions and classes
- **PEP 8 Compliance**: Code follows Python style guidelines
- **Error Handling**: Proper exception handling with meaningful messages

### Example of Improved Code Style

**Before:**
```python
def draw_pool_table(ax, table, style='standard', show_pockets=True):
    # Draw table
    pass
```

**After:**
```python
def draw_pool_table(
    ax: matplotlib.axes.Axes, 
    table: PoolTable,  
    style: str = 'standard', 
    show_pockets: bool = True
) -> Dict[str, Any]:
    \"\"\"
    Draw a standardized pool table with consistent styling.
    
    Args:
        ax: Matplotlib axis to draw on.
        table: PoolTable object containing dimensions.
        style: Visual style - 'standard', 'compact', or 'minimal'.
        show_pockets: Whether to draw pocket circles.
        
    Returns:
        Dictionary containing styling information for ball rendering.
        
    Raises:
        ValueError: If style parameter is not recognized.
    \"\"\"
    # Implementation with proper validation
```

## Running Visualizations

### Break Shot Analysis
```bash
# Single rack type
python visualizations/visualize_break_improved.py 9-ball 12.0

# All rack types
python visualizations/visualize_break_improved.py all 15.0

# Time-series analysis
python visualizations/visualize_break_timelapse.py 8-ball 12.0 6.0 0.25
```

### Rack Visualization
```bash
python visualizations/visualize_rack.py 9-ball
```

## Key Features

### Enhanced Ball Tracking
The visualization system now properly tracks balls that:
- Fall into pockets (y < table.H - ball_radius)
- Escape through table boundaries (|x| > table.W/2 + ball_radius or |z| > table.L/2 + ball_radius)
- Are unused in 9-ball/10-ball games

### Pocketed Ball Display
Missing balls are now shown in a line below the table with:
- Proper ball colors and numbers
- "Pocketed" label for clarity
- Consistent styling across all visualizations

### Modular Design
- `table_renderer.py`: Centralized rendering functions
- Consistent styling across all visualizations
- Configurable display options (pockets, spots, grid, pocketed area)

## Development Workflow

### Adding New Features
1. Write code in appropriate directory (`visualizations/`, `utils/`, etc.)
2. Add proper type hints and docstrings
3. Include error handling and validation
4. Update tests if needed
5. Test with multiple rack types and speeds

### Running Tests
```bash
# Run all tests
pytest

# Run specific test
python tests/test_rack_arrangement.py
```

### Debugging Tools
Use the debug tools for analysis:
```bash
python debug_tools/debug_ball_positions.py
python debug_tools/analyze_segments.py
```

## Code Organization Benefits

1. **Separation of Concerns**: Visualization, physics, and utilities are separated
2. **Reusability**: Common functions in `utils/` can be shared
3. **Maintainability**: Clear structure makes code easy to modify
4. **Testing**: Isolated test files for different components
5. **Documentation**: Comprehensive docstrings and type hints
6. **Extensibility**: Easy to add new visualization types or analysis tools

## Migration from Old Structure

Old files have been moved as follows:
- `visualize_*.py` → `visualizations/`
- `test_*.py` → `tests/`
- `debug_*.py`, `analyze_*.py` → `debug_tools/`
- `table_renderer.py` → `utils/`

Import statements have been updated to work with the new structure.