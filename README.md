# PoolPhysics
Event-based billiards physics simulator (decoupled from https://github.com/jzitelli/poolvr.py)

The `pool_physics` package implements an event-based pool physics simulator based on the paper:
```
  AN EVENT-BASED POOL PHYSICS SIMULATOR
  Will Leckie, Michael Greenspan
  DOI: 10.1007/11922155_19 · Source: DBLP
  Conference: Advances in Computer Games, 11th International Conference,
  Taipei, Taiwan, September 6-9, 2005.
```

## Installation & Setup

### Quick Setup
```bash
# Create and activate virtual environment
./scripts/build_virtual_environment.sh
source venv313/bin/activate  # or venv3XX/bin/activate for your Python version

# Install dependencies
pip install -r requirements.txt

# Build shared libraries (Linux)
./scripts/build_shared_libs_linux.sh

# Build package
./scripts/build_package.sh
```

### Windows Setup
```bash
# Build shared libraries (Windows)
./scripts/build_shared_libs_windows.sh
```

## Available Commands

### 🎯 Quick Start & Examples

#### Run All Examples
```bash
python run_examples.py
```
Automated runner that executes visualization examples and demonstrates code quality improvements.

#### Enhanced Physics Demo
```bash
python demo_enhanced_physics.py
```
Comprehensive demonstration showcasing fixed X-direction rail collision detection and physics parameter testing.

#### Rail Fix Demo
```bash
python demo_rail_fix.py
```
Specific demonstration of the rail collision fixes.

### 🎱 Visualizations

#### Break Shot Visualizations
```bash
# Basic break visualization with comparison
python visualizations/visualize_break.py [rack_type] [break_speed] [spacing_mode] [seed]
# Examples:
python visualizations/visualize_break.py 8-ball 12
python visualizations/visualize_break.py 9-ball 15 random 42
python visualizations/visualize_break.py 10-ball 10 tight
```

#### Improved Break Visualization
```bash
# Enhanced visualization with pocketed balls display
python visualizations/visualize_break_improved.py [rack_type] [break_speed] [spacing_mode] [seed]
# Examples:
python visualizations/visualize_break_improved.py 9-ball 12
python visualizations/visualize_break_improved.py 8-ball 14 fixed
```

#### Break Timelapse
```bash
# Time-series snapshots of ball movement
python visualizations/visualize_break_timelapse.py [rack_type] [break_speed] [num_snapshots] [time_interval] [spacing_mode] [seed]
# Examples:
python visualizations/visualize_break_timelapse.py 8-ball 12 4 0.5
python visualizations/visualize_break_timelapse.py 9-ball 15 6 0.3 random 123
```

#### Rack Visualization
```bash
# Visualize different rack arrangements
python visualizations/visualize_rack.py [rack_type] [spacing_mode] [seed]
# Examples:
python visualizations/visualize_rack.py 8-ball
python visualizations/visualize_rack.py 9-ball tight
```

**Parameters:**
- `rack_type`: `8-ball`, `9-ball`, or `10-ball`
- `break_speed`: Cue ball speed in m/s (typically 8-20)
- `spacing_mode`: `random`, `fixed`, or `tight`
- `seed`: Integer for reproducible random spacing
- `num_snapshots`: Number of time snapshots (default: 4)
- `time_interval`: Time between snapshots in seconds (default: 0.5)

### 📊 Analysis Tools

#### Break Variations Analysis
```bash
# Multi-run statistical analysis of break shots
python analyze/analyze_break_variations.py [rack_type] [num_runs] [break_speed] [spacing_mode]
# Examples:
python analyze/analyze_break_variations.py 8-ball 50 12 random
python analyze/analyze_break_variations.py 9-ball 100 15 fixed
```

#### Rack Arrangement Testing
```bash
# Test different rack arrangements and spacing
python analyze/test_rack_arrangement.py
```

#### Rail Collision Testing
```bash
# Test rail collision detection in all directions
python analyze/test_rail_collision.py
```

### 🔧 Debug Tools

#### Physics Parameter Testing
```bash
# Comprehensive physics parameter validation
python debug_tools/physics_parameter_tester.py
```

#### Rail Bounce Testing
```bash
# Quick rail collision validation
python debug_tools/quick_rail_test.py
python debug_tools/debug_rail_bounce.py
```

#### Collision Algorithm Analysis
```bash
# Detailed collision detection analysis
python debug_tools/debug_collision_algorithm.py
python debug_tools/comprehensive_physics_validation.py
```

#### Ball Position Debugging
```bash
# Debug ball positioning and movement
python debug_tools/debug_ball_positions.py
python debug_tools/debug_cue_ball_escape.py
```

#### Event System Debugging
```bash
# Analyze physics events and timing
python debug_tools/debug_events.py
python debug_tools/show_physics_data.py
```

#### Segment Analysis
```bash
# Analyze table segments and collision boundaries
python debug_tools/analyze_segments.py
python debug_tools/debug_segments_detailed.py
```

### 🧪 Testing

#### Run All Tests
```bash
# Run pytest suite
pytest

# Run specific test files
pytest test/test_physics.py
pytest test/test_collisions.py
pytest test/test_poly_solvers.py
```

#### Individual Test Categories
```bash
# Physics engine tests
python -m pytest test/test_physics.py -v

# Collision detection tests  
python -m pytest test/test_collisions.py -v

# Polynomial solver tests
python -m pytest test/test_poly_solvers.py -v
```

## Output Files

All visualizations are saved to the `images/` directory with descriptive filenames including parameters and timestamps.

**Example outputs:**
- `images/8-ball_break_comparison_speed12.0_random_seed42.png`
- `images/9-ball_break_timelapse_speed15.0_snapshots4.png`
- `images/rack_arrangement_10-ball_tight.png`

## Project Structure

```
PoolPhysics/
├── pool_physics/          # Core physics engine
├── visualizations/        # All visualization scripts
├── utils/                 # Shared utility functions  
├── test/                  # Pytest-based test suite
├── analyze/               # Analysis and debugging tools
├── debug_tools/           # Additional debug utilities
├── scripts/               # Build and setup scripts
├── images/                # Generated outputs
└── DEVELOPMENT_GUIDE.md   # Documentation
```

## Physics Features

- **Fixed X-direction rail collision detection**  
- **Comprehensive physics parameter testing framework**  
- **Multiple rack types** (8-ball, 9-ball, 10-ball)  
- **Configurable ball spacing** (random, fixed, tight)  
- **Reproducible simulations** with seed support  
- **Time-series analysis** and visualization  
- **Statistical break analysis** with multi-run support  
- **Extensive debugging tools** for physics validation
