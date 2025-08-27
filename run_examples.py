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


def main() -> None:
    """Main function to run visualization examples."""
    print("🎱 Pool Physics - Example Runner")
    print("=" * 40)
    print("This script runs example visualizations to demonstrate the physics engine.")
    print("For complete documentation, see README.md")
    
    print("\n" + "=" * 40)
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