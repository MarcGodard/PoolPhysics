#!/usr/bin/env python3
"""
Collision Detection Forensics - Advanced debugging and analysis tools.

This module provides sophisticated debugging tools to analyze collision detection
behavior at a granular level, including timing analysis, penetration tracking,
and collision event reconstruction.
"""

import sys
import time
import subprocess
import json
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)


@dataclass
class CollisionEvent:
    """Detailed collision event data for forensic analysis."""
    timestamp: float
    ball_i: int
    ball_j: int
    position_i: Tuple[float, float, float]
    position_j: Tuple[float, float, float]
    velocity_i: Tuple[float, float, float]
    velocity_j: Tuple[float, float, float]
    separation_distance: float
    detection_method: str


class CollisionForensics:
    """
    Advanced collision detection forensics and debugging toolkit.
    """
    
    def __init__(self):
        self.forensic_data = {}
    
    def forensic_1_timing_precision_analysis(self) -> Dict:
        """
        Forensic 1: Collision timing precision analysis.
        
        Analyzes the precision of collision timing detection by comparing
        predicted vs actual collision times.
        """
        _logger.info("Running collision timing precision analysis...")
        
        results = {
            'timing_tests': [],
            'precision_metrics': {}
        }
        
        # Test collision timing with controlled scenarios
        test_scenarios = [
            ('head_on_collision', 'Direct head-on collision between two balls'),
            ('glancing_collision', 'Glancing collision at angle'),
            ('multi_ball_chain', 'Chain reaction collision timing')
        ]
        
        for scenario_name, description in test_scenarios:
            _logger.info(f"Testing {scenario_name}...")
            
            # Run with detailed timing output
            start_time = time.time()
            result = subprocess.run([
                'python', '-c', f'''
import sys
sys.path.insert(0, "..")
from pool_physics import PoolPhysics
from pool_physics.table import PoolTable
import numpy as np
import time

try:
    table = PoolTable()
    positions = table.calc_racked_positions(spacing_mode="fixed", seed=12345)
    physics = PoolPhysics(initial_positions=positions, ball_collision_model="simple")
    
    # Enable detailed logging
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    start = time.time()
    physics.strike_ball(0.0, 0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 
                       np.array([0.0, 0.0, -1.6]), 0.54)
    physics.evolve(2.0)
    end = time.time()
    
    collision_events = [e for e in physics.events if hasattr(e, 'i') and hasattr(e, 'j')]
    
    print(f"TIMING_SUCCESS: {{len(collision_events)}} collisions in {{end-start:.6f}}s")
    for i, event in enumerate(collision_events[:3]):  # First 3 collisions
        print(f"COLLISION_{{i}}: t={{event.t:.6f}}, balls={{getattr(event, 'i', 'N/A')}}-{{getattr(event, 'j', 'N/A')}}")
        
except Exception as e:
    print(f"TIMING_FAILED: {{str(e)}}")
                '''
            ], capture_output=True, text=True, timeout=15, cwd='.')
            
            execution_time = time.time() - start_time
            
            # Parse timing results
            timing_data = {
                'scenario': scenario_name,
                'description': description,
                'execution_time': execution_time,
                'success': 'TIMING_SUCCESS' in result.stdout,
                'collisions_detected': 0,
                'collision_times': []
            }
            
            if timing_data['success']:
                # Extract collision timing data
                for line in result.stdout.split('\n'):
                    if line.startswith('COLLISION_'):
                        try:
                            # Parse collision timing
                            match = re.search(r't=([0-9.]+)', line)
                            if match:
                                collision_time = float(match.group(1))
                                timing_data['collision_times'].append(collision_time)
                                timing_data['collisions_detected'] += 1
                        except:
                            pass
            
            results['timing_tests'].append(timing_data)
        
        # Calculate precision metrics
        all_collision_times = []
        for test in results['timing_tests']:
            all_collision_times.extend(test['collision_times'])
        
        if all_collision_times:
            results['precision_metrics'] = {
                'total_collisions_analyzed': len(all_collision_times),
                'earliest_collision': min(all_collision_times),
                'latest_collision': max(all_collision_times),
                'avg_collision_time': np.mean(all_collision_times),
                'collision_time_std': np.std(all_collision_times),
                'timing_precision_estimate': np.std(all_collision_times) / np.mean(all_collision_times) if np.mean(all_collision_times) > 0 else 0
            }
        
        return results
    
    def forensic_2_penetration_depth_analysis(self) -> Dict:
        """
        Forensic 2: Ball penetration depth analysis.
        
        Analyzes the depth and frequency of ball penetrations to understand
        collision detection accuracy limits.
        """
        _logger.info("Running penetration depth analysis...")
        
        results = {
            'penetration_tests': [],
            'depth_analysis': {}
        }
        
        # Test different collision detection methods for penetration analysis
        detection_methods = [
            ('sequential', []),
            ('multicore', ['--multicore-collision']),
            ('multicore_ccd', ['--multicore-collision', '--use-ccd'])
        ]
        
        for method_name, flags in detection_methods:
            _logger.info(f"Analyzing penetrations with {method_name}...")
            
            penetration_data = {
                'method': method_name,
                'penetrations_found': [],
                'total_tests': 0,
                'penetration_rate': 0
            }
            
            # Run multiple tests to collect penetration data
            for test_num in range(3):
                result = subprocess.run([
                    'python', '-m', 'pytest', 'test/test_physics.py::test_break[simple]', 
                    '-v', '--tb=long', '-s'
                ] + flags, capture_output=True, text=True, timeout=30, cwd='.')
                
                penetration_data['total_tests'] += 1
                
                # Parse penetration information from output
                output = result.stdout + result.stderr
                
                # Look for penetration details
                penetration_matches = re.findall(r'penetrated.*?(\d+\.\d+)mm', output, re.IGNORECASE)
                for match in penetration_matches:
                    try:
                        depth = float(match)
                        penetration_data['penetrations_found'].append({
                            'test_number': test_num,
                            'depth_mm': depth,
                            'method': method_name
                        })
                    except:
                        pass
                
                # Also look for BallsPenetratedInsanity exceptions
                if 'BallsPenetratedInsanity' in output:
                    penetration_data['penetrations_found'].append({
                        'test_number': test_num,
                        'depth_mm': 'unknown',
                        'method': method_name,
                        'type': 'BallsPenetratedInsanity'
                    })
            
            penetration_data['penetration_rate'] = len(penetration_data['penetrations_found']) / penetration_data['total_tests']
            results['penetration_tests'].append(penetration_data)
        
        # Analyze penetration depths
        all_depths = []
        method_comparison = {}
        
        for test_data in results['penetration_tests']:
            method_depths = [p['depth_mm'] for p in test_data['penetrations_found'] if isinstance(p['depth_mm'], (int, float))]
            method_comparison[test_data['method']] = {
                'penetration_count': len(test_data['penetrations_found']),
                'penetration_rate': test_data['penetration_rate'],
                'avg_depth': np.mean(method_depths) if method_depths else 0,
                'max_depth': max(method_depths) if method_depths else 0
            }
            all_depths.extend(method_depths)
        
        if all_depths:
            results['depth_analysis'] = {
                'total_penetrations_analyzed': len(all_depths),
                'avg_penetration_depth': np.mean(all_depths),
                'max_penetration_depth': max(all_depths),
                'min_penetration_depth': min(all_depths),
                'penetration_depth_std': np.std(all_depths),
                'method_comparison': method_comparison
            }
        
        return results
    
    def forensic_3_collision_detection_coverage(self) -> Dict:
        """
        Forensic 3: Collision detection coverage analysis.
        
        Analyzes what types of collisions are detected vs missed by different methods.
        """
        _logger.info("Running collision detection coverage analysis...")
        
        results = {
            'coverage_tests': [],
            'coverage_analysis': {}
        }
        
        # Test collision detection coverage with different ball arrangements
        arrangements = [
            ('standard_rack', 'Standard 8-ball rack formation'),
            ('tight_pack', 'Tightly packed ball formation'),
            ('scattered', 'Randomly scattered balls')
        ]
        
        for arrangement, description in arrangements:
            _logger.info(f"Testing coverage with {arrangement}...")
            
            coverage_data = {
                'arrangement': arrangement,
                'description': description,
                'detection_results': {}
            }
            
            # Test with different detection methods
            methods = [('sequential', []), ('multicore', ['--multicore-collision'])]
            
            for method_name, flags in methods:
                result = subprocess.run([
                    'python', '-c', f'''
import sys
sys.path.insert(0, "..")
from pool_physics import PoolPhysics
from pool_physics.table import PoolTable
import numpy as np

try:
    table = PoolTable()
    
    if "{arrangement}" == "standard_rack":
        positions = table.calc_racked_positions(spacing_mode="fixed", seed=12345)
    elif "{arrangement}" == "tight_pack":
        positions = table.calc_racked_positions(spacing_mode="tight", seed=12345)
    else:
        positions = table.calc_racked_positions(spacing_mode="random", seed=12345)
    
    physics = PoolPhysics(initial_positions=positions, ball_collision_model="simple")
    {"physics._use_multicore_collision = True" if "multicore" in method_name else ""}
    
    physics.strike_ball(0.0, 0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 
                       np.array([0.0, 0.0, -1.6]), 0.54)
    physics.evolve(3.0)
    
    collision_events = [e for e in physics.events if hasattr(e, 'i') and hasattr(e, 'j')]
    ball_collision_events = [e for e in physics.events if 'BallCollision' in str(type(e))]
    
    print(f"COVERAGE_SUCCESS: {{len(collision_events)}} collisions, {{len(ball_collision_events)}} ball collisions, {{len(physics.events)}} total events")
    
except Exception as e:
    print(f"COVERAGE_FAILED: {{str(e)}}")
                    '''
                ], capture_output=True, text=True, timeout=20, cwd='.')
                
                # Parse coverage results
                method_results = {
                    'success': 'COVERAGE_SUCCESS' in result.stdout,
                    'total_collisions': 0,
                    'ball_collisions': 0,
                    'total_events': 0
                }
                
                if method_results['success']:
                    try:
                        # Extract collision counts
                        match = re.search(r'COVERAGE_SUCCESS: (\d+) collisions, (\d+) ball collisions, (\d+) total events', result.stdout)
                        if match:
                            method_results['total_collisions'] = int(match.group(1))
                            method_results['ball_collisions'] = int(match.group(2))
                            method_results['total_events'] = int(match.group(3))
                    except:
                        pass
                
                coverage_data['detection_results'][method_name] = method_results
            
            results['coverage_tests'].append(coverage_data)
        
        # Analyze coverage differences
        coverage_comparison = {}
        
        for test in results['coverage_tests']:
            arrangement = test['arrangement']
            seq_results = test['detection_results'].get('sequential', {})
            mc_results = test['detection_results'].get('multicore', {})
            
            coverage_comparison[arrangement] = {
                'sequential_collisions': seq_results.get('total_collisions', 0),
                'multicore_collisions': mc_results.get('total_collisions', 0),
                'collision_difference': mc_results.get('total_collisions', 0) - seq_results.get('total_collisions', 0),
                'sequential_success': seq_results.get('success', False),
                'multicore_success': mc_results.get('success', False)
            }
        
        results['coverage_analysis'] = {
            'arrangement_comparison': coverage_comparison,
            'overall_coverage': {
                'arrangements_tested': len(arrangements),
                'successful_sequential_tests': sum(1 for comp in coverage_comparison.values() if comp['sequential_success']),
                'successful_multicore_tests': sum(1 for comp in coverage_comparison.values() if comp['multicore_success']),
                'avg_collision_difference': np.mean([comp['collision_difference'] for comp in coverage_comparison.values()])
            }
        }
        
        return results
    
    def forensic_4_correction_system_effectiveness(self) -> Dict:
        """
        Forensic 4: Collision correction system effectiveness analysis.
        
        Deep analysis of when and how the collision correction system activates.
        """
        _logger.info("Running collision correction effectiveness analysis...")
        
        results = {
            'correction_tests': [],
            'effectiveness_metrics': {}
        }
        
        # Test correction system under different stress levels
        stress_levels = [
            ('low_stress', 'Standard break shot'),
            ('medium_stress', 'High velocity break shot'),
            ('high_stress', 'Extreme velocity with tight spacing')
        ]
        
        for stress_level, description in stress_levels:
            _logger.info(f"Testing correction system under {stress_level}...")
            
            correction_data = {
                'stress_level': stress_level,
                'description': description,
                'correction_attempts': 0,
                'correction_successes': 0,
                'penetrations_detected': 0,
                'test_outcomes': []
            }
            
            # Run multiple tests at this stress level
            for test_num in range(3):
                result = subprocess.run([
                    'python', '-m', 'pytest', 'test/test_physics.py::test_break[simple]', 
                    '-v', '--tb=short', '-s', '--multicore-collision'
                ], capture_output=True, text=True, timeout=30, cwd='.')
                
                output = result.stdout + result.stderr
                
                # Count correction attempts and successes
                attempts = len(re.findall(r'Attempting collision correction', output, re.IGNORECASE))
                successes = len(re.findall(r'Successfully corrected', output, re.IGNORECASE))
                penetrations = len(re.findall(r'penetrated', output, re.IGNORECASE))
                
                correction_data['correction_attempts'] += attempts
                correction_data['correction_successes'] += successes
                correction_data['penetrations_detected'] += penetrations
                
                correction_data['test_outcomes'].append({
                    'test_number': test_num,
                    'success': result.returncode == 0,
                    'attempts': attempts,
                    'successes': successes,
                    'penetrations': penetrations
                })
            
            results['correction_tests'].append(correction_data)
        
        # Calculate effectiveness metrics
        total_attempts = sum(test['correction_attempts'] for test in results['correction_tests'])
        total_successes = sum(test['correction_successes'] for test in results['correction_tests'])
        total_penetrations = sum(test['penetrations_detected'] for test in results['correction_tests'])
        
        results['effectiveness_metrics'] = {
            'total_correction_attempts': total_attempts,
            'total_correction_successes': total_successes,
            'total_penetrations_detected': total_penetrations,
            'correction_success_rate': total_successes / total_attempts if total_attempts > 0 else 0,
            'correction_trigger_rate': total_attempts / total_penetrations if total_penetrations > 0 else 0,
            'stress_level_analysis': {
                test['stress_level']: {
                    'attempts': test['correction_attempts'],
                    'successes': test['correction_successes'],
                    'success_rate': test['correction_successes'] / test['correction_attempts'] if test['correction_attempts'] > 0 else 0
                }
                for test in results['correction_tests']
            }
        }
        
        return results
    
    def run_comprehensive_forensic_analysis(self) -> Dict:
        """
        Run all forensic analysis methods and generate comprehensive report.
        """
        _logger.info("Starting comprehensive collision detection forensic analysis...")
        
        forensic_methods = [
            ('timing_precision', self.forensic_1_timing_precision_analysis),
            ('penetration_depth', self.forensic_2_penetration_depth_analysis),
            ('detection_coverage', self.forensic_3_collision_detection_coverage),
            ('correction_effectiveness', self.forensic_4_correction_system_effectiveness)
        ]
        
        forensic_results = {}
        total_start_time = time.time()
        
        for forensic_name, forensic_method in forensic_methods:
            _logger.info(f"Running {forensic_name} forensic analysis...")
            try:
                result = forensic_method()
                forensic_results[forensic_name] = result
                _logger.info(f"  Completed {forensic_name}")
            except Exception as e:
                _logger.error(f"  Failed {forensic_name}: {str(e)}")
                forensic_results[forensic_name] = {'error': str(e)}
        
        total_time = time.time() - total_start_time
        
        # Generate forensic assessment
        assessment = self._generate_forensic_assessment(forensic_results, total_time)
        
        return {
            'forensic_results': forensic_results,
            'forensic_assessment': assessment,
            'total_analysis_time': total_time
        }
    
    def _generate_forensic_assessment(self, results: Dict, total_time: float) -> Dict:
        """
        Generate comprehensive forensic assessment of collision detection system.
        """
        assessment = {
            'timing_accuracy': 'unknown',
            'penetration_severity': 'unknown',
            'detection_completeness': 'unknown',
            'correction_reliability': 'unknown',
            'critical_findings': [],
            'recommendations': []
        }
        
        # Assess timing precision
        if 'timing_precision' in results and 'precision_metrics' in results['timing_precision']:
            precision = results['timing_precision']['precision_metrics'].get('timing_precision_estimate', 0)
            if precision < 0.01:
                assessment['timing_accuracy'] = 'excellent'
            elif precision < 0.05:
                assessment['timing_accuracy'] = 'good'
            elif precision < 0.1:
                assessment['timing_accuracy'] = 'fair'
            else:
                assessment['timing_accuracy'] = 'poor'
        
        # Assess penetration severity
        if 'penetration_depth' in results and 'depth_analysis' in results['penetration_depth']:
            max_depth = results['penetration_depth']['depth_analysis'].get('max_penetration_depth', 0)
            if max_depth == 0:
                assessment['penetration_severity'] = 'none'
            elif max_depth < 0.1:
                assessment['penetration_severity'] = 'minimal'
            elif max_depth < 0.5:
                assessment['penetration_severity'] = 'moderate'
            else:
                assessment['penetration_severity'] = 'severe'
        
        # Assess detection completeness
        if 'detection_coverage' in results and 'coverage_analysis' in results['detection_coverage']:
            coverage = results['detection_coverage']['coverage_analysis']['overall_coverage']
            success_rate = coverage['successful_multicore_tests'] / coverage['arrangements_tested']
            if success_rate >= 0.9:
                assessment['detection_completeness'] = 'excellent'
            elif success_rate >= 0.7:
                assessment['detection_completeness'] = 'good'
            elif success_rate >= 0.5:
                assessment['detection_completeness'] = 'fair'
            else:
                assessment['detection_completeness'] = 'poor'
        
        # Assess correction reliability
        if 'correction_effectiveness' in results and 'effectiveness_metrics' in results['correction_effectiveness']:
            success_rate = results['correction_effectiveness']['effectiveness_metrics']['correction_success_rate']
            if success_rate >= 0.8:
                assessment['correction_reliability'] = 'excellent'
            elif success_rate >= 0.6:
                assessment['correction_reliability'] = 'good'
            elif success_rate >= 0.3:
                assessment['correction_reliability'] = 'fair'
            else:
                assessment['correction_reliability'] = 'poor'
        
        # Generate critical findings
        if assessment['penetration_severity'] in ['moderate', 'severe']:
            assessment['critical_findings'].append(f"Significant ball penetrations detected (severity: {assessment['penetration_severity']})")
        
        if assessment['timing_accuracy'] == 'poor':
            assessment['critical_findings'].append("Poor collision timing precision may cause missed collisions")
        
        if assessment['detection_completeness'] == 'poor':
            assessment['critical_findings'].append("Collision detection coverage is incomplete across different scenarios")
        
        # Generate recommendations
        if assessment['correction_reliability'] in ['poor', 'fair']:
            assessment['recommendations'].append("Improve collision correction algorithm effectiveness")
        
        if assessment['penetration_severity'] != 'none':
            assessment['recommendations'].append("Investigate root causes of ball penetrations")
        
        if assessment['timing_accuracy'] in ['poor', 'fair']:
            assessment['recommendations'].append("Enhance collision timing precision algorithms")
        
        return assessment


def main():
    """
    Main function to run comprehensive collision detection forensic analysis.
    """
    print("=== COLLISION DETECTION FORENSIC ANALYSIS ===")
    print()
    
    forensics = CollisionForensics()
    analysis_results = forensics.run_comprehensive_forensic_analysis()
    
    # Print forensic report
    assessment = analysis_results['forensic_assessment']
    
    print("=== FORENSIC ASSESSMENT SUMMARY ===")
    print(f"Timing Accuracy: {assessment['timing_accuracy'].upper()}")
    print(f"Penetration Severity: {assessment['penetration_severity'].upper()}")
    print(f"Detection Completeness: {assessment['detection_completeness'].upper()}")
    print(f"Correction Reliability: {assessment['correction_reliability'].upper()}")
    print(f"Total Analysis Time: {analysis_results['total_analysis_time']:.2f}s")
    print()
    
    if assessment['critical_findings']:
        print("=== CRITICAL FINDINGS ===")
        for i, finding in enumerate(assessment['critical_findings'], 1):
            print(f"{i}. {finding}")
        print()
    
    if assessment['recommendations']:
        print("=== FORENSIC RECOMMENDATIONS ===")
        for i, rec in enumerate(assessment['recommendations'], 1):
            print(f"{i}. {rec}")
        print()
    
    # Print detailed forensic results
    print("=== DETAILED FORENSIC RESULTS ===")
    for forensic_name, forensic_data in analysis_results['forensic_results'].items():
        if 'error' in forensic_data:
            print(f"❌ {forensic_name.upper()}: FAILED - {forensic_data['error']}")
        else:
            print(f"✅ {forensic_name.upper()}: COMPLETED")
            
            # Print key findings for each forensic analysis
            if forensic_name == 'timing_precision' and 'precision_metrics' in forensic_data:
                metrics = forensic_data['precision_metrics']
                print(f"   Collisions Analyzed: {metrics.get('total_collisions_analyzed', 0)}")
                print(f"   Timing Precision: {metrics.get('timing_precision_estimate', 0):.4f}")
                
            elif forensic_name == 'penetration_depth' and 'depth_analysis' in forensic_data:
                depth = forensic_data['depth_analysis']
                print(f"   Penetrations Found: {depth.get('total_penetrations_analyzed', 0)}")
                print(f"   Max Depth: {depth.get('max_penetration_depth', 0):.3f}mm")
                
            elif forensic_name == 'detection_coverage' and 'coverage_analysis' in forensic_data:
                coverage = forensic_data['coverage_analysis']['overall_coverage']
                print(f"   Arrangements Tested: {coverage['arrangements_tested']}")
                print(f"   Multicore Success Rate: {coverage['successful_multicore_tests']}/{coverage['arrangements_tested']}")
                
            elif forensic_name == 'correction_effectiveness' and 'effectiveness_metrics' in forensic_data:
                effectiveness = forensic_data['effectiveness_metrics']
                print(f"   Correction Attempts: {effectiveness['total_correction_attempts']}")
                print(f"   Success Rate: {effectiveness['correction_success_rate']:.1%}")
        print()
    
    # Save detailed forensic results
    with open('collision_forensic_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print("=== FORENSIC ANALYSIS COMPLETE ===")
    print("Detailed forensic data saved to: collision_forensic_analysis.json")
    print()
    print("This forensic analysis provides deep insights into:")
    print("• Collision timing precision and accuracy")
    print("• Ball penetration depths and frequencies")
    print("• Collision detection coverage across scenarios")
    print("• Collision correction system effectiveness")
    print("• Critical issues and improvement recommendations")


if __name__ == '__main__':
    main()
