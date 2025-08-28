#!/usr/bin/env python3
"""
Advanced collision detection benchmarking and deep testing strategies.

This module implements multiple sophisticated testing approaches to thoroughly
validate collision detection systems using existing working test infrastructure.
"""

import sys
import time
import subprocess
import random
import numpy as np
import json
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)


class CollisionBenchmarkSuite:
    """
    Comprehensive collision detection benchmarking using existing test infrastructure.
    """
    
    def __init__(self):
        self.results = {}
        self.lock = threading.Lock()
    
    def benchmark_1_sequential_vs_multicore_accuracy(self) -> Dict:
        """
        Benchmark 1: Sequential vs Multicore collision detection accuracy comparison.
        
        Runs identical scenarios with both methods to identify accuracy differences.
        """
        _logger.info("Running Sequential vs Multicore accuracy benchmark...")
        
        results = {
            'sequential': {'passes': 0, 'fails': 0, 'times': [], 'penetrations': []},
            'multicore': {'passes': 0, 'fails': 0, 'times': [], 'penetrations': []},
            'accuracy_comparison': {}
        }
        
        # Test with multiple collision models and scenarios
        test_scenarios = [
            ('simple', 'test_break[simple]'),
            ('simple', 'test_break[simulated]'),
        ]
        
        for model, test_name in test_scenarios:
            _logger.info(f"Testing {model} model with {test_name}")
            
            # Run sequential baseline
            for i in range(3):
                start_time = time.time()
                result = subprocess.run([
                    'python', '-m', 'pytest', f'test/test_physics.py::{test_name}', 
                    '-v', '--tb=short', '-q'
                ], capture_output=True, text=True, timeout=30, cwd='.')
                
                exec_time = time.time() - start_time
                results['sequential']['times'].append(exec_time)
                
                if result.returncode == 0:
                    results['sequential']['passes'] += 1
                else:
                    results['sequential']['fails'] += 1
                    # Extract penetration info
                    if 'penetrated' in result.stdout.lower():
                        results['sequential']['penetrations'].append(result.stdout)
            
            # Run multicore comparison
            for i in range(3):
                start_time = time.time()
                result = subprocess.run([
                    'python', '-m', 'pytest', f'test/test_physics.py::{test_name}', 
                    '-v', '--tb=short', '-q', '--multicore-collision'
                ], capture_output=True, text=True, timeout=30, cwd='.')
                
                exec_time = time.time() - start_time
                results['multicore']['times'].append(exec_time)
                
                if result.returncode == 0:
                    results['multicore']['passes'] += 1
                else:
                    results['multicore']['fails'] += 1
                    # Extract penetration info
                    if 'penetrated' in result.stdout.lower():
                        results['multicore']['penetrations'].append(result.stdout)
        
        # Calculate accuracy metrics
        seq_accuracy = results['sequential']['passes'] / (results['sequential']['passes'] + results['sequential']['fails'])
        mc_accuracy = results['multicore']['passes'] / (results['multicore']['passes'] + results['multicore']['fails'])
        
        results['accuracy_comparison'] = {
            'sequential_accuracy': seq_accuracy,
            'multicore_accuracy': mc_accuracy,
            'accuracy_difference': mc_accuracy - seq_accuracy,
            'sequential_avg_time': np.mean(results['sequential']['times']),
            'multicore_avg_time': np.mean(results['multicore']['times']),
            'speedup': np.mean(results['sequential']['times']) / np.mean(results['multicore']['times'])
        }
        
        return results
    
    def benchmark_2_collision_detection_sensitivity(self) -> Dict:
        """
        Benchmark 2: Collision detection sensitivity analysis.
        
        Tests how sensitive collision detection is to different ball arrangements.
        """
        _logger.info("Running collision detection sensitivity analysis...")
        
        results = {
            'spacing_modes': {},
            'sensitivity_metrics': {}
        }
        
        spacing_modes = ['random', 'fixed', 'tight']
        
        for spacing in spacing_modes:
            _logger.info(f"Testing {spacing} spacing mode...")
            
            mode_results = {'passes': 0, 'fails': 0, 'times': [], 'penetrations': 0}
            
            # Run multiple tests with this spacing mode
            for i in range(5):
                start_time = time.time()
                result = subprocess.run([
                    'python', '-c', f'''
import sys
sys.path.insert(0, "..")
from pool_physics import PoolPhysics
from pool_physics.table import PoolTable
import numpy as np

try:
    table = PoolTable()
    positions = table.calc_racked_positions(spacing_mode="{spacing}", seed={i+1000})
    physics = PoolPhysics(initial_positions=positions, ball_collision_model="simple")
    physics._use_multicore_collision = True
    
    # Standard break shot
    physics.strike_ball(0.0, 0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 
                       np.array([0.0, 0.0, -1.6]), 0.54)
    physics.evolve(3.0)
    
    print(f"SUCCESS: {{len(physics.events)}} events generated")
except Exception as e:
    print(f"FAILED: {{str(e)}}")
                    '''
                ], capture_output=True, text=True, timeout=15, cwd='.')
                
                exec_time = time.time() - start_time
                mode_results['times'].append(exec_time)
                
                if 'SUCCESS' in result.stdout:
                    mode_results['passes'] += 1
                else:
                    mode_results['fails'] += 1
                    if 'penetrated' in result.stdout.lower():
                        mode_results['penetrations'] += 1
            
            results['spacing_modes'][spacing] = mode_results
        
        # Calculate sensitivity metrics
        total_tests = sum(data['passes'] + data['fails'] for data in results['spacing_modes'].values())
        total_passes = sum(data['passes'] for data in results['spacing_modes'].values())
        
        results['sensitivity_metrics'] = {
            'overall_success_rate': total_passes / total_tests if total_tests > 0 else 0,
            'spacing_reliability': {
                mode: data['passes'] / (data['passes'] + data['fails']) 
                for mode, data in results['spacing_modes'].items()
            },
            'penetration_rates': {
                mode: data['penetrations'] / (data['passes'] + data['fails'])
                for mode, data in results['spacing_modes'].items()
            }
        }
        
        return results
    
    def benchmark_3_performance_scaling_analysis(self) -> Dict:
        """
        Benchmark 3: Performance scaling with different core counts.
        
        Tests how collision detection performance scales with CPU core usage.
        """
        _logger.info("Running performance scaling analysis...")
        
        results = {
            'core_scaling': {},
            'scaling_metrics': {}
        }
        
        # Test with different core counts
        core_counts = [1, 2, 4, 8]
        
        for cores in core_counts:
            _logger.info(f"Testing with {cores} cores...")
            
            core_results = {'times': [], 'passes': 0, 'fails': 0}
            
            for i in range(3):
                start_time = time.time()
                result = subprocess.run([
                    'python', '-m', 'pytest', 'test/test_physics.py::test_break[simple]', 
                    '-v', '--tb=no', '-q', '--multicore-collision', f'--max-cores={cores}'
                ], capture_output=True, text=True, timeout=30, cwd='.')
                
                exec_time = time.time() - start_time
                core_results['times'].append(exec_time)
                
                if result.returncode == 0:
                    core_results['passes'] += 1
                else:
                    core_results['fails'] += 1
            
            results['core_scaling'][cores] = core_results
        
        # Calculate scaling efficiency
        baseline_time = np.mean(results['core_scaling'][1]['times']) if 1 in results['core_scaling'] else None
        
        if baseline_time:
            results['scaling_metrics'] = {
                'speedups': {
                    cores: baseline_time / np.mean(data['times']) if data['times'] else 0
                    for cores, data in results['core_scaling'].items()
                },
                'efficiency': {
                    cores: (baseline_time / np.mean(data['times'])) / cores if data['times'] else 0
                    for cores, data in results['core_scaling'].items()
                }
            }
        
        return results
    
    def benchmark_4_collision_correction_stress_test(self) -> Dict:
        """
        Benchmark 4: Collision correction system stress testing.
        
        Tests collision correction under high-stress scenarios.
        """
        _logger.info("Running collision correction stress test...")
        
        results = {
            'correction_scenarios': {},
            'correction_effectiveness': {}
        }
        
        # Test different collision detection configurations
        configs = [
            ('multicore', ['--multicore-collision']),
            ('multicore_ccd', ['--multicore-collision', '--use-ccd'])
        ]
        
        for config_name, flags in configs:
            _logger.info(f"Testing {config_name} configuration...")
            
            config_results = {
                'passes': 0, 'fails': 0, 'times': [],
                'penetrations': 0, 'corrections': 0
            }
            
            for i in range(5):
                start_time = time.time()
                result = subprocess.run([
                    'python', '-m', 'pytest', 'test/test_physics.py::test_break[simple]', 
                    '-v', '--tb=short', '-s'
                ] + flags, capture_output=True, text=True, timeout=30, cwd='.')
                
                exec_time = time.time() - start_time
                config_results['times'].append(exec_time)
                
                if result.returncode == 0:
                    config_results['passes'] += 1
                else:
                    config_results['fails'] += 1
                    
                    # Analyze output for correction activity
                    output = result.stdout + result.stderr
                    if 'penetrated' in output.lower():
                        config_results['penetrations'] += 1
                    if 'correction' in output.lower():
                        config_results['corrections'] += 1
            
            results['correction_scenarios'][config_name] = config_results
        
        # Calculate correction effectiveness
        total_penetrations = sum(data['penetrations'] for data in results['correction_scenarios'].values())
        total_corrections = sum(data['corrections'] for data in results['correction_scenarios'].values())
        
        results['correction_effectiveness'] = {
            'total_penetrations_detected': total_penetrations,
            'total_correction_attempts': total_corrections,
            'correction_trigger_rate': total_corrections / total_penetrations if total_penetrations > 0 else 0,
            'config_reliability': {
                config: data['passes'] / (data['passes'] + data['fails'])
                for config, data in results['correction_scenarios'].items()
            }
        }
        
        return results
    
    def benchmark_5_monte_carlo_reliability_test(self) -> Dict:
        """
        Benchmark 5: Monte Carlo reliability testing with statistical analysis.
        
        Runs many random scenarios to build statistical confidence in collision detection.
        """
        _logger.info("Running Monte Carlo reliability test...")
        
        results = {
            'monte_carlo_results': [],
            'statistical_analysis': {}
        }
        
        num_iterations = 20
        
        def run_single_test(iteration):
            """Run a single Monte Carlo test iteration."""
            try:
                start_time = time.time()
                result = subprocess.run([
                    'python', '-c', f'''
import sys, random
sys.path.insert(0, "..")
from pool_physics import PoolPhysics
from pool_physics.table import PoolTable
import numpy as np

random.seed({iteration + 5000})
np.random.seed({iteration + 5000})

try:
    table = PoolTable()
    positions = table.calc_racked_positions(spacing_mode="random", seed={iteration + 5000})
    physics = PoolPhysics(initial_positions=positions, ball_collision_model="simple")
    physics._use_multicore_collision = True
    
    # Random strike parameters
    velocity = np.array([
        random.uniform(-0.05, 0.05),
        0.0,
        random.uniform(-2.0, -1.2)
    ])
    
    physics.strike_ball(0.0, 0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 
                       velocity, 0.54)
    physics.evolve(3.0)
    
    print(f"SUCCESS: {{len(physics.events)}} events, {{len([e for e in physics.events if hasattr(e, 'i')])}} collisions")
except Exception as e:
    print(f"FAILED: {{str(e)}}")
                    '''
                ], capture_output=True, text=True, timeout=20, cwd='.')
                
                exec_time = time.time() - start_time
                
                success = 'SUCCESS' in result.stdout
                events = 0
                collisions = 0
                
                if success:
                    # Extract metrics from output
                    try:
                        output_parts = result.stdout.split()
                        events = int(output_parts[1])
                        collisions = int(output_parts[4])
                    except:
                        pass
                
                return {
                    'iteration': iteration,
                    'success': success,
                    'execution_time': exec_time,
                    'events': events,
                    'collisions': collisions,
                    'error': None if success else result.stdout
                }
                
            except Exception as e:
                return {
                    'iteration': iteration,
                    'success': False,
                    'execution_time': 0,
                    'events': 0,
                    'collisions': 0,
                    'error': str(e)
                }
        
        # Run tests in parallel for speed
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_single_test, i) for i in range(num_iterations)]
            
            for future in as_completed(futures):
                result = future.result()
                results['monte_carlo_results'].append(result)
                
                status = "PASS" if result['success'] else "FAIL"
                _logger.info(f"  Iteration {result['iteration']}: {status} ({result['execution_time']:.2f}s)")
        
        # Statistical analysis
        successful_tests = [r for r in results['monte_carlo_results'] if r['success']]
        failed_tests = [r for r in results['monte_carlo_results'] if not r['success']]
        
        success_rate = len(successful_tests) / len(results['monte_carlo_results'])
        
        if successful_tests:
            avg_time = np.mean([r['execution_time'] for r in successful_tests])
            std_time = np.std([r['execution_time'] for r in successful_tests])
            avg_events = np.mean([r['events'] for r in successful_tests])
            avg_collisions = np.mean([r['collisions'] for r in successful_tests])
        else:
            avg_time = std_time = avg_events = avg_collisions = 0
        
        results['statistical_analysis'] = {
            'total_iterations': num_iterations,
            'successful_tests': len(successful_tests),
            'failed_tests': len(failed_tests),
            'success_rate': success_rate,
            'confidence_interval_95': success_rate * 1.96 * np.sqrt(success_rate * (1 - success_rate) / num_iterations),
            'performance_stats': {
                'avg_execution_time': avg_time,
                'std_execution_time': std_time,
                'avg_events_per_test': avg_events,
                'avg_collisions_per_test': avg_collisions
            },
            'failure_analysis': {
                'common_errors': list(set([r['error'] for r in failed_tests if r['error']]))[:5]
            }
        }
        
        return results
    
    def run_comprehensive_benchmark_suite(self) -> Dict:
        """
        Run all benchmarking tests and generate comprehensive analysis.
        """
        _logger.info("Starting comprehensive collision detection benchmark suite...")
        
        benchmark_methods = [
            ('accuracy_comparison', self.benchmark_1_sequential_vs_multicore_accuracy),
            ('sensitivity_analysis', self.benchmark_2_collision_detection_sensitivity),
            ('performance_scaling', self.benchmark_3_performance_scaling_analysis),
            ('correction_stress_test', self.benchmark_4_collision_correction_stress_test),
            ('monte_carlo_reliability', self.benchmark_5_monte_carlo_reliability_test)
        ]
        
        suite_results = {}
        total_start_time = time.time()
        
        for benchmark_name, benchmark_method in benchmark_methods:
            _logger.info(f"Running {benchmark_name} benchmark...")
            try:
                result = benchmark_method()
                suite_results[benchmark_name] = result
                _logger.info(f"  Completed {benchmark_name}")
            except Exception as e:
                _logger.error(f"  Failed {benchmark_name}: {str(e)}")
                suite_results[benchmark_name] = {'error': str(e)}
        
        total_time = time.time() - total_start_time
        
        # Generate overall assessment
        assessment = self._generate_overall_assessment(suite_results, total_time)
        
        return {
            'benchmark_results': suite_results,
            'overall_assessment': assessment,
            'total_execution_time': total_time
        }
    
    def _generate_overall_assessment(self, results: Dict, total_time: float) -> Dict:
        """
        Generate overall assessment of collision detection system.
        """
        assessment = {
            'system_reliability': 'unknown',
            'performance_rating': 'unknown',
            'correction_effectiveness': 'unknown',
            'recommendations': []
        }
        
        # Analyze Monte Carlo results for reliability
        if 'monte_carlo_reliability' in results:
            mc_results = results['monte_carlo_reliability']
            if 'statistical_analysis' in mc_results:
                success_rate = mc_results['statistical_analysis']['success_rate']
                
                if success_rate >= 0.9:
                    assessment['system_reliability'] = 'excellent'
                elif success_rate >= 0.8:
                    assessment['system_reliability'] = 'good'
                elif success_rate >= 0.6:
                    assessment['system_reliability'] = 'fair'
                else:
                    assessment['system_reliability'] = 'poor'
        
        # Analyze performance scaling
        if 'performance_scaling' in results:
            scaling_results = results['performance_scaling']
            if 'scaling_metrics' in scaling_results and 'speedups' in scaling_results['scaling_metrics']:
                max_speedup = max(scaling_results['scaling_metrics']['speedups'].values())
                
                if max_speedup >= 4.0:
                    assessment['performance_rating'] = 'excellent'
                elif max_speedup >= 2.0:
                    assessment['performance_rating'] = 'good'
                elif max_speedup >= 1.2:
                    assessment['performance_rating'] = 'fair'
                else:
                    assessment['performance_rating'] = 'poor'
        
        # Analyze correction effectiveness
        if 'correction_stress_test' in results:
            correction_results = results['correction_stress_test']
            if 'correction_effectiveness' in correction_results:
                trigger_rate = correction_results['correction_effectiveness']['correction_trigger_rate']
                
                if trigger_rate >= 0.8:
                    assessment['correction_effectiveness'] = 'excellent'
                elif trigger_rate >= 0.6:
                    assessment['correction_effectiveness'] = 'good'
                elif trigger_rate >= 0.3:
                    assessment['correction_effectiveness'] = 'fair'
                else:
                    assessment['correction_effectiveness'] = 'poor'
        
        # Generate recommendations
        if assessment['system_reliability'] == 'poor':
            assessment['recommendations'].append("Investigate fundamental collision detection algorithm issues")
        
        if assessment['performance_rating'] == 'poor':
            assessment['recommendations'].append("Optimize multicore collision detection implementation")
        
        if assessment['correction_effectiveness'] == 'poor':
            assessment['recommendations'].append("Improve collision correction trigger sensitivity and effectiveness")
        
        return assessment


def main():
    """
    Main function to run comprehensive collision detection benchmarking.
    """
    print("=== COMPREHENSIVE COLLISION DETECTION BENCHMARKING ===")
    print()
    
    benchmark_suite = CollisionBenchmarkSuite()
    results = benchmark_suite.run_comprehensive_benchmark_suite()
    
    # Print comprehensive report
    print("=== BENCHMARK RESULTS SUMMARY ===")
    print(f"Total Execution Time: {results['total_execution_time']:.2f}s")
    print()
    
    assessment = results['overall_assessment']
    print("=== SYSTEM ASSESSMENT ===")
    print(f"System Reliability: {assessment['system_reliability'].upper()}")
    print(f"Performance Rating: {assessment['performance_rating'].upper()}")
    print(f"Correction Effectiveness: {assessment['correction_effectiveness'].upper()}")
    print()
    
    if assessment['recommendations']:
        print("=== RECOMMENDATIONS ===")
        for i, rec in enumerate(assessment['recommendations'], 1):
            print(f"{i}. {rec}")
        print()
    
    # Print detailed results for each benchmark
    for benchmark_name, benchmark_data in results['benchmark_results'].items():
        if 'error' in benchmark_data:
            print(f"❌ {benchmark_name.upper()}: FAILED - {benchmark_data['error']}")
        else:
            print(f"✅ {benchmark_name.upper()}: COMPLETED")
            
            # Print key metrics for each benchmark
            if benchmark_name == 'monte_carlo_reliability' and 'statistical_analysis' in benchmark_data:
                stats = benchmark_data['statistical_analysis']
                print(f"   Success Rate: {stats['success_rate']:.1%}")
                print(f"   Avg Time: {stats['performance_stats']['avg_execution_time']:.3f}s")
                
            elif benchmark_name == 'accuracy_comparison' and 'accuracy_comparison' in benchmark_data:
                acc = benchmark_data['accuracy_comparison']
                print(f"   Sequential Accuracy: {acc['sequential_accuracy']:.1%}")
                print(f"   Multicore Accuracy: {acc['multicore_accuracy']:.1%}")
                print(f"   Speedup: {acc['speedup']:.1f}x")
                
            elif benchmark_name == 'performance_scaling' and 'scaling_metrics' in benchmark_data:
                if 'speedups' in benchmark_data['scaling_metrics']:
                    max_speedup = max(benchmark_data['scaling_metrics']['speedups'].values())
                    print(f"   Max Speedup: {max_speedup:.1f}x")
        print()
    
    # Save detailed results
    with open('collision_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("=== DEEP TESTING STRATEGIES IMPLEMENTED ===")
    print("1. ✅ Sequential vs Multicore accuracy comparison")
    print("2. ✅ Collision detection sensitivity analysis")
    print("3. ✅ Performance scaling with different core counts")
    print("4. ✅ Collision correction stress testing")
    print("5. ✅ Monte Carlo reliability testing with statistical analysis")
    print()
    print("Detailed results saved to: collision_benchmark_results.json")
    print("=== COMPREHENSIVE BENCHMARKING COMPLETE ===")


if __name__ == '__main__':
    main()
