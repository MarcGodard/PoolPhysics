#!/usr/bin/env python3
"""
Large-scale collision detection validation for 8-ball break scenarios.

This script runs extensive tests with random break positions to validate
the collision detection and correction systems at scale.
"""

import sys
import time
import random
import numpy as np
import logging
from typing import List, Dict, Tuple
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pool_physics import PoolPhysics
from pool_physics.table import PoolTable
from pool_physics.events import PhysicsEvent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)


class CollisionTestSuite:
    """
    Comprehensive collision detection test suite for random 8-ball breaks.
    """
    
    def __init__(self, num_tests: int = 100, enable_multicore: bool = True, 
                 enable_ccd: bool = False, enable_correction: bool = True):
        self.num_tests = num_tests
        self.enable_multicore = enable_multicore
        self.enable_ccd = enable_ccd
        self.enable_correction = enable_correction
        
        # Test results tracking
        self.results = {
            'total_tests': 0,
            'successful_tests': 0,
            'penetration_detected': 0,
            'corrections_attempted': 0,
            'corrections_successful': 0,
            'execution_times': [],
            'failed_tests': [],
            'penetration_details': []
        }
        
        # Create table for ball positioning
        self.table = PoolTable(ball_radius=PhysicsEvent.ball_radius)
    
    def run_single_test(self, test_id: int, seed: int = None) -> Dict:
        """
        Run a single collision detection test with random break position.
        """
        if seed is None:
            seed = random.randint(1, 1000000)
        
        test_result = {
            'test_id': test_id,
            'seed': seed,
            'success': False,
            'execution_time': 0,
            'penetrations': 0,
            'corrections_attempted': 0,
            'corrections_successful': 0,
            'error': None
        }
        
        try:
            # Create random ball positions
            positions = self.table.calc_racked_positions(spacing_mode='random', seed=seed)
            
            # Create physics engine
            physics = PoolPhysics(initial_positions=positions, ball_collision_model='simple')
            
            # Configure collision detection systems
            if self.enable_multicore:
                physics._use_multicore_collision = True
                
                if hasattr(physics, '_multicore_detector'):
                    if self.enable_ccd:
                        physics._multicore_detector.enable_ccd(True)
            
            # Track collision correction if enabled
            correction_stats = {'attempts': 0, 'successes': 0}
            if self.enable_correction and hasattr(physics, '_collision_corrector'):
                original_check = physics._collision_corrector.check_and_correct_penetrations
                
                def tracked_check(current_time):
                    correction_stats['attempts'] += 1
                    result = original_check(current_time)
                    if result:
                        correction_stats['successes'] += 1
                    return result
                
                physics._collision_corrector.check_and_correct_penetrations = tracked_check
            
            # Execute break shot
            start_time = time.time()
            
            # Strike cue ball with random parameters
            strike_velocity = np.array([-0.01 + random.uniform(-0.005, 0.005), 0.0, 
                                      -1.6 + random.uniform(-0.2, 0.2)])
            physics.strike_ball(0.0, 0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 
                              strike_velocity, 0.54)
            
            # Evolve physics for 5 seconds
            physics.evolve(5.0)
            
            execution_time = time.time() - start_time
            
            # Test completed successfully
            test_result.update({
                'success': True,
                'execution_time': execution_time,
                'corrections_attempted': correction_stats['attempts'],
                'corrections_successful': correction_stats['successes'],
                'num_events': len(physics.events)
            })
            
        except Exception as e:
            test_result['error'] = str(e)
            
            # Check if it's a penetration error
            if 'penetrated' in str(e).lower() or 'BallsPenetratedInsanity' in str(e):
                test_result['penetrations'] = 1
                
                # Extract penetration details if available
                if hasattr(e, 'args') and len(e.args) > 0:
                    error_msg = str(e.args[0])
                    test_result['penetration_details'] = error_msg
        
        return test_result
    
    def run_test_suite(self) -> Dict:
        """
        Run the complete test suite with random break positions.
        """
        _logger.info(f"Starting large-scale collision test suite:")
        _logger.info(f"  Tests: {self.num_tests}")
        _logger.info(f"  Multicore: {self.enable_multicore}")
        _logger.info(f"  CCD: {self.enable_ccd}")
        _logger.info(f"  Correction: {self.enable_correction}")
        
        start_time = time.time()
        
        for test_id in range(1, self.num_tests + 1):
            _logger.info(f"Running test {test_id}/{self.num_tests}...")
            
            result = self.run_single_test(test_id)
            
            # Update statistics
            self.results['total_tests'] += 1
            
            if result['success']:
                self.results['successful_tests'] += 1
                self.results['execution_times'].append(result['execution_time'])
            else:
                self.results['failed_tests'].append(result)
                
                if result['penetrations'] > 0:
                    self.results['penetration_detected'] += 1
                    self.results['penetration_details'].append({
                        'test_id': test_id,
                        'seed': result['seed'],
                        'details': result.get('penetration_details', 'No details')
                    })
            
            self.results['corrections_attempted'] += result['corrections_attempted']
            self.results['corrections_successful'] += result['corrections_successful']
            
            # Progress update every 10 tests
            if test_id % 10 == 0:
                success_rate = (self.results['successful_tests'] / self.results['total_tests']) * 100
                _logger.info(f"  Progress: {test_id}/{self.num_tests} ({success_rate:.1f}% success rate)")
        
        total_time = time.time() - start_time
        self.results['total_execution_time'] = total_time
        
        return self.results
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive test report.
        """
        results = self.results
        
        # Calculate statistics
        success_rate = (results['successful_tests'] / results['total_tests']) * 100
        penetration_rate = (results['penetration_detected'] / results['total_tests']) * 100
        
        avg_time = np.mean(results['execution_times']) if results['execution_times'] else 0
        std_time = np.std(results['execution_times']) if results['execution_times'] else 0
        
        correction_success_rate = 0
        if results['corrections_attempted'] > 0:
            correction_success_rate = (results['corrections_successful'] / results['corrections_attempted']) * 100
        
        report = f"""
=== LARGE-SCALE COLLISION DETECTION TEST REPORT ===

Configuration:
  Total Tests: {results['total_tests']}
  Multicore Enabled: {self.enable_multicore}
  CCD Enabled: {self.enable_ccd}
  Correction Enabled: {self.enable_correction}

Results Summary:
  Successful Tests: {results['successful_tests']}/{results['total_tests']} ({success_rate:.1f}%)
  Failed Tests: {len(results['failed_tests'])} ({100-success_rate:.1f}%)
  Penetrations Detected: {results['penetration_detected']} ({penetration_rate:.1f}%)

Performance:
  Total Execution Time: {results['total_execution_time']:.2f}s
  Average Test Time: {avg_time:.3f}s ± {std_time:.3f}s
  Tests per Second: {results['total_tests']/results['total_execution_time']:.1f}

Collision Correction:
  Correction Attempts: {results['corrections_attempted']}
  Successful Corrections: {results['corrections_successful']}
  Correction Success Rate: {correction_success_rate:.1f}%

System Assessment:
"""
        
        if success_rate >= 95:
            report += "  ✓ EXCELLENT: Collision detection system highly reliable\n"
        elif success_rate >= 85:
            report += "  ✓ GOOD: Collision detection system mostly reliable\n"
        elif success_rate >= 70:
            report += "  ⚠ FAIR: Collision detection system needs improvement\n"
        else:
            report += "  ✗ POOR: Collision detection system requires major fixes\n"
        
        if penetration_rate == 0:
            report += "  ✓ PERFECT: Zero penetrations detected\n"
        elif penetration_rate < 5:
            report += "  ✓ EXCELLENT: Very low penetration rate\n"
        elif penetration_rate < 15:
            report += "  ⚠ ACCEPTABLE: Moderate penetration rate\n"
        else:
            report += "  ✗ HIGH: Significant penetration issues\n"
        
        if correction_success_rate >= 90:
            report += "  ✓ EXCELLENT: Collision correction highly effective\n"
        elif correction_success_rate >= 70:
            report += "  ✓ GOOD: Collision correction mostly effective\n"
        elif correction_success_rate >= 50:
            report += "  ⚠ FAIR: Collision correction partially effective\n"
        else:
            report += "  ✗ POOR: Collision correction needs improvement\n"
        
        # Add failed test details if any
        if results['failed_tests']:
            report += f"\nFailed Test Details:\n"
            for i, failed_test in enumerate(results['failed_tests'][:5]):  # Show first 5
                report += f"  Test {failed_test['test_id']} (seed={failed_test['seed']}): {failed_test['error']}\n"
            
            if len(results['failed_tests']) > 5:
                report += f"  ... and {len(results['failed_tests']) - 5} more failed tests\n"
        
        return report


def main():
    """
    Main function to run large-scale collision detection tests.
    """
    # Test configurations to run
    test_configs = [
        {
            'name': 'Sequential Baseline',
            'multicore': False,
            'ccd': False,
            'correction': False,
            'num_tests': 50
        },
        {
            'name': 'Multicore + Adaptive',
            'multicore': True,
            'ccd': False,
            'correction': True,
            'num_tests': 100
        },
        {
            'name': 'Multicore + CCD',
            'multicore': True,
            'ccd': True,
            'correction': True,
            'num_tests': 100
        }
    ]
    
    all_reports = []
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"RUNNING: {config['name']}")
        print(f"{'='*60}")
        
        test_suite = CollisionTestSuite(
            num_tests=config['num_tests'],
            enable_multicore=config['multicore'],
            enable_ccd=config['ccd'],
            enable_correction=config['correction']
        )
        
        results = test_suite.run_test_suite()
        report = test_suite.generate_report()
        
        print(report)
        all_reports.append(f"=== {config['name']} ===\n{report}")
    
    # Save comprehensive report
    with open('collision_test_report.txt', 'w') as f:
        f.write('\n\n'.join(all_reports))
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE COLLISION TESTING COMPLETE")
    print("Report saved to: collision_test_report.txt")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
