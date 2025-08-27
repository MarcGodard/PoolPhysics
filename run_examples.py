#!/usr/bin/env python3
"""
Example runner script demonstrating the cleaned-up Pool Physics codebase.

This script showcases the organized structure and improved code quality with
proper type hints, documentation, and error handling.
"""

import os
import sys
import subprocess
from typing import List, Dict, Any


def run_visualization_examples() -> None:
    """Run example visualizations to demonstrate the cleaned-up codebase."""
    print("🎱 Pool Physics - Clean Code Examples")
    print("=" * 50)
    
    # Check if virtual environment exists
    venv_python = os.path.join('venv313', 'bin', 'python')
    if not os.path.exists(venv_python):
        print("❌ Virtual environment not found at venv313/bin/python")
        print("   Please run: python -m venv venv313 && source venv313/bin/activate && pip install -r requirements.txt")
        return
    
    examples = [
        {
            'name': '9-Ball Break Visualization',
            'script': 'visualizations/visualize_break_improved.py',
            'args': ['9-ball', '12'],
            'description': 'Shows before/after comparison with pocketed balls displayed below table'
        },
        {
            'name': '8-Ball Break Timelapse',  
            'script': 'visualizations/visualize_break_timelapse.py',
            'args': ['8-ball', '12', '4', '0.5'],
            'description': 'Time-series snapshots showing ball movement over time'
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}")
        print(f"   Description: {example['description']}")
        print(f"   Command: {venv_python} {example['script']} {' '.join(example['args'])}")
        
        # Check if script exists
        if not os.path.exists(example['script']):
            print(f"   ❌ Script not found: {example['script']}")
            continue
        
        try:
            result = subprocess.run(
                [venv_python, example['script']] + example['args'],
                capture_output=True, 
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("   ✅ Successfully generated visualization")
                # Print key stats from output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'pocketed' in line.lower() or 'saved as' in line.lower():
                        print(f"   📊 {line.strip()}")
            else:
                print(f"   ❌ Error: {result.stderr.strip()}")
                
        except subprocess.TimeoutExpired:
            print("   ⏱️  Timeout - visualization may still be processing")
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    print(f"\n📁 Generated visualizations saved to: {os.path.abspath('images/')}")


def show_code_quality_improvements() -> None:
    """Demonstrate code quality improvements."""
    print("\n🔧 Code Quality Improvements")
    print("=" * 30)
    
    improvements = [
        "✅ Proper directory structure (visualizations/, utils/, tests/, debug_tools/)",
        "✅ Type hints for all function parameters and return values", 
        "✅ Comprehensive docstrings with Args, Returns, and Raises sections",
        "✅ Input validation and error handling",
        "✅ Consistent code formatting following PEP 8",
        "✅ Modular design with reusable utility functions",
        "✅ Enhanced ball tracking (pocketed + escaped balls)",
        "✅ Centralized table rendering with configurable styles"
    ]
    
    for improvement in improvements:
        print(f"   {improvement}")


def show_directory_structure() -> None:
    """Show the cleaned-up directory structure."""
    print("\n📁 Organized Directory Structure")  
    print("=" * 35)
    
    structure = """
    PoolPhysics/
    ├── pool_physics/          # Core physics engine
    ├── visualizations/        # All visualization scripts
    ├── utils/                 # Shared utility functions  
    ├── test/                  # Pytest-based test suite
    ├── analyze/               # Analysis and debugging tools
    ├── debug_tools/           # Additional debug utilities
    ├── images/                # Generated outputs
    └── DEVELOPMENT_GUIDE.md   # Documentation
    """
    
    print(structure)


def main() -> None:
    """Main function demonstrating the cleaned-up codebase."""
    show_directory_structure()
    show_code_quality_improvements()
    
    print("\n" + "=" * 50)
    response = input("Run visualization examples? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        run_visualization_examples()
    else:
        print("\n👍 Skipping visualizations. You can run them manually using:")
        print("   python visualizations/visualize_break_improved.py 9-ball 12")
        print("   python visualizations/visualize_break_timelapse.py 8-ball")
        print("   python visualizations/visualize_rack.py 10-ball")


if __name__ == "__main__":
    main()