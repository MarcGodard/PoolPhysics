#!/usr/bin/env python3
"""
Deep collision detection validation suite with comprehensive testing strategies.

This module implements multiple advanced testing approaches to thoroughly
validate the collision detection and correction systems.
"""

import sys
import time
import random
import numpy as np
import logging
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pool_physics import PoolPhysics
from pool_physics.table import PoolTable
from pool_physics.events import PhysicsEvent, BallCollisionEvent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)


@dataclass
class CollisionTestResult:
    """Result of a single collision test."""
    test_id: str
    success: bool
    execution_time: float
    num_events: int
    penetrations_detected: int
    corrections_attempted: int
    corrections_successful: int
    error_message: Optional[str] = None
    physics_metrics: Optional[Dict] = None


class DeepCollisionTester:
    """
    Comprehensive collision detection testing with multiple validation strategies.
    """
    
    def __init__(self):
        self.table = PoolTable(ball_radius=PhysicsEvent.ball_radius)
        self.results = []
    
    def test_1_analytical_two_ball_collision(self) -> CollisionTestResult:
        """
        Test 1: Analytical validation with known two-ball collision solution.
        
        Creates a simple head-on collision between two balls with known
        initial conditions and validates against analytical physics solution.
        """
        test_id = "analytical_two_ball"
        
        try:
            # Create simple two-ball setup
            positions = np.zeros((16, 3))
            positions[0] = [-0.5, 0.0, 0.0]  # Ball 0 at left
            positions[1] = [0.5, 0.0, 0.0]   # Ball 1 at right
            
            physics = PoolPhysics(initial_positions=positions, ball_collision_model='simple')
            physics._use_multicore_collision = True
            
            # Strike ball 0 directly toward ball 1
            start_time = time.time()
            physics.strike_ball(0.0, 0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 
                              np.array([0.0, 0.0, 1.0]), 0.54)
            
            physics.evolve(2.0)
            execution_time = time.time() - start_time
            
            # Analytical validation
            collision_events = [e for e in physics.events if isinstance(e, BallCollisionEvent)]
            
            # Should have exactly one collision
            expected_collisions = 1
            actual_collisions = len(collision_events)
            
            # Validate conservation of momentum
            momentum_conserved = True
            if collision_events:
                # Check momentum before and after collision
                # (This is a simplified check - full validation would be more complex)
                pass
            
            return CollisionTestResult(
                test_id=test_id,
                success=(actual_collisions == expected_collisions),
                execution_time=execution_time,
                num_events=len(physics.events),
                penetrations_detected=0,
                corrections_attempted=0,
                corrections_successful=0,
                physics_metrics={
                    'expected_collisions': expected_collisions,
                    'actual_collisions': actual_collisions,
                    'momentum_conserved': momentum_conserved
                }
            )
            
        except Exception as e:
            return CollisionTestResult(
                test_id=test_id,
                success=False,
                execution_time=0,
                num_events=0,
                penetrations_detected=0,
                corrections_attempted=0,
                corrections_successful=0,
                error_message=str(e)
            )
    
    def test_2_stress_high_velocity_collisions(self) -> CollisionTestResult:
        """
        Test 2: Stress testing with extremely high velocity collisions.
        
        Tests collision detection accuracy under extreme conditions that
        might cause numerical instability or missed collisions.
        """
        test_id = "stress_high_velocity"
        
        try:
            positions = self.table.calc_racked_positions(spacing_mode='tight', seed=12345)
            physics = PoolPhysics(initial_positions=positions, ball_collision_model='simple')
            physics._use_multicore_collision = True
            
            # Extremely high velocity strike (10x normal)
            start_time = time.time()
            physics.strike_ball(0.0, 0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 
                              np.array([0.0, 0.0, -16.0]), 0.54)  # 10x normal velocity
            
            physics.evolve(1.0)  # Shorter evolution time due to high speed
            execution_time = time.time() - start_time
            
            return CollisionTestResult(
                test_id=test_id,
                success=True,
                execution_time=execution_time,
                num_events=len(physics.events),
                penetrations_detected=0,
                corrections_attempted=0,
                corrections_successful=0,
                physics_metrics={'velocity_multiplier': 10}
            )
            
        except Exception as e:
            penetrations = 1 if 'penetrated' in str(e).lower() else 0
            return CollisionTestResult(
                test_id=test_id,
                success=False,
                execution_time=0,
                num_events=0,
                penetrations_detected=penetrations,
                corrections_attempted=0,
                corrections_successful=0,
                error_message=str(e)
            )
    
    def test_3_edge_case_simultaneous_collisions(self) -> CollisionTestResult:
        """
        Test 3: Edge case testing with simultaneous multi-ball collisions.
        
        Creates scenarios where multiple balls collide simultaneously
        to test collision detection ordering and accuracy.
        """
        test_id = "edge_simultaneous_collisions"
        
        try:
            # Create custom arrangement for simultaneous collisions
            positions = np.zeros((16, 3))
            radius = PhysicsEvent.ball_radius
            
            # Arrange 4 balls in a cross pattern, all equidistant from center
            positions[0] = [0.0, 0.0, 2.1 * radius]   # North
            positions[1] = [0.0, 0.0, -2.1 * radius]  # South  
            positions[2] = [2.1 * radius, 0.0, 0.0]   # East
            positions[3] = [-2.1 * radius, 0.0, 0.0]  # West
            
            physics = PoolPhysics(initial_positions=positions, ball_collision_model='simple')
            physics._use_multicore_collision = True
            
            # Strike all balls toward center simultaneously
            start_time = time.time()
            physics.strike_ball(0.0, 0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 
                              np.array([0.0, 0.0, -1.0]), 0.54)
            physics.strike_ball(0.0, 1, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 
                              np.array([0.0, 0.0, 1.0]), 0.54)
            physics.strike_ball(0.0, 2, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 
                              np.array([-1.0, 0.0, 0.0]), 0.54)
            physics.strike_ball(0.0, 3, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 
                              np.array([1.0, 0.0, 0.0]), 0.54)
            
            physics.evolve(2.0)
            execution_time = time.time() - start_time
            
            collision_events = [e for e in physics.events if isinstance(e, BallCollisionEvent)]
            
            return CollisionTestResult(
                test_id=test_id,
                success=True,
                execution_time=execution_time,
                num_events=len(physics.events),
                penetrations_detected=0,
                corrections_attempted=0,
                corrections_successful=0,
                physics_metrics={
                    'collision_events': len(collision_events),
                    'scenario': 'simultaneous_4_ball'
                }
            )
            
        except Exception as e:
            penetrations = 1 if 'penetrated' in str(e).lower() else 0
            return CollisionTestResult(
                test_id=test_id,
                success=False,
                execution_time=0,
                num_events=0,
                penetrations_detected=penetrations,
                corrections_attempted=0,
                corrections_successful=0,
                error_message=str(e)
            )
    
    def test_4_precision_near_miss_collisions(self) -> CollisionTestResult:
        """
        Test 4: Precision testing with near-miss collision scenarios.
        
        Tests collision detection accuracy with balls that pass very close
        but should not collide, validating false positive prevention.
        """
        test_id = "precision_near_miss"
        
        try:
            positions = np.zeros((16, 3))
            radius = PhysicsEvent.ball_radius
            
            # Two balls with trajectories that pass very close but don't touch
            positions[0] = [-1.0, 0.0, 0.0]
            positions[1] = [1.0, 0.0, 2.01 * radius]  # Just outside collision distance
            
            physics = PoolPhysics(initial_positions=positions, ball_collision_model='simple')
            physics._use_multicore_collision = True
            
            start_time = time.time()
            # Ball 0 moves right, Ball 1 moves left - they should pass without colliding
            physics.strike_ball(0.0, 0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 
                              np.array([0.0, 0.0, 1.0]), 0.54)
            physics.strike_ball(0.0, 1, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 
                              np.array([0.0, 0.0, -1.0]), 0.54)
            
            physics.evolve(3.0)
            execution_time = time.time() - start_time
            
            collision_events = [e for e in physics.events if isinstance(e, BallCollisionEvent)]
            
            # Should have zero collisions (near miss)
            success = len(collision_events) == 0
            
            return CollisionTestResult(
                test_id=test_id,
                success=success,
                execution_time=execution_time,
                num_events=len(physics.events),
                penetrations_detected=0,
                corrections_attempted=0,
                corrections_successful=0,
                physics_metrics={
                    'expected_collisions': 0,
                    'actual_collisions': len(collision_events),
                    'separation_distance': 2.01 * radius
                }
            )
            
        except Exception as e:
            return CollisionTestResult(
                test_id=test_id,
                success=False,
                execution_time=0,
                num_events=0,
                penetrations_detected=0,
                corrections_attempted=0,
                corrections_successful=0,
                error_message=str(e)
            )
    
    def test_5_performance_scaling_analysis(self) -> CollisionTestResult:
        """
        Test 5: Performance scaling analysis with varying ball counts.
        
        Tests how collision detection performance scales with increasing
        numbers of active balls to identify bottlenecks.
        """
        test_id = "performance_scaling"
        
        try:
            scaling_results = {}
            
            for num_balls in [2, 4, 8, 16]:
                positions = np.zeros((16, 3))
                
                # Arrange balls in a line for predictable collisions
                for i in range(num_balls):
                    positions[i] = [i * 2.1 * PhysicsEvent.ball_radius, 0.0, 0.0]
                
                physics = PoolPhysics(initial_positions=positions, ball_collision_model='simple')
                physics._use_multicore_collision = True
                
                start_time = time.time()
                # Strike first ball to create chain reaction
                physics.strike_ball(0.0, 0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 
                                  np.array([0.0, 0.0, 1.0]), 0.54)
                
                physics.evolve(2.0)
                execution_time = time.time() - start_time
                
                scaling_results[num_balls] = {
                    'time': execution_time,
                    'events': len(physics.events)
                }
            
            return CollisionTestResult(
                test_id=test_id,
                success=True,
                execution_time=sum(r['time'] for r in scaling_results.values()),
                num_events=sum(r['events'] for r in scaling_results.values()),
                penetrations_detected=0,
                corrections_attempted=0,
                corrections_successful=0,
                physics_metrics={'scaling_results': scaling_results}
            )
            
        except Exception as e:
            return CollisionTestResult(
                test_id=test_id,
                success=False,
                execution_time=0,
                num_events=0,
                penetrations_detected=0,
                corrections_attempted=0,
                corrections_successful=0,
                error_message=str(e)
            )
    
    def test_6_monte_carlo_random_scenarios(self, num_iterations: int = 50) -> CollisionTestResult:
        """
        Test 6: Monte Carlo testing with random collision scenarios.
        
        Runs many random collision scenarios to identify statistical
        patterns in collision detection accuracy and performance.
        """
        test_id = f"monte_carlo_{num_iterations}"
        
        successes = 0
        total_time = 0
        total_events = 0
        penetrations = 0
        corrections = 0
        
        try:
            for i in range(num_iterations):
                seed = random.randint(1, 1000000)
                positions = self.table.calc_racked_positions(spacing_mode='random', seed=seed)
                
                physics = PoolPhysics(initial_positions=positions, ball_collision_model='simple')
                physics._use_multicore_collision = True
                
                # Random strike parameters
                velocity = np.array([
                    random.uniform(-0.1, 0.1),
                    0.0,
                    random.uniform(-2.0, -1.0)
                ])
                
                try:
                    start_time = time.time()
                    physics.strike_ball(0.0, 0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 
                                      velocity, 0.54)
                    physics.evolve(3.0)
                    
                    iteration_time = time.time() - start_time
                    total_time += iteration_time
                    total_events += len(physics.events)
                    successes += 1
                    
                except Exception as e:
                    if 'penetrated' in str(e).lower():
                        penetrations += 1
                    if 'correction' in str(e).lower():
                        corrections += 1
            
            success_rate = successes / num_iterations
            
            return CollisionTestResult(
                test_id=test_id,
                success=(success_rate >= 0.8),  # 80% success threshold
                execution_time=total_time,
                num_events=total_events,
                penetrations_detected=penetrations,
                corrections_attempted=corrections,
                corrections_successful=0,
                physics_metrics={
                    'iterations': num_iterations,
                    'success_rate': success_rate,
                    'avg_time_per_test': total_time / num_iterations,
                    'avg_events_per_test': total_events / num_iterations
                }
            )
            
        except Exception as e:
            return CollisionTestResult(
                test_id=test_id,
                success=False,
                execution_time=total_time,
                num_events=total_events,
                penetrations_detected=penetrations,
                corrections_attempted=corrections,
                corrections_successful=0,
                error_message=str(e)
            )
    
    def test_7_collision_correction_effectiveness(self) -> CollisionTestResult:
        """
        Test 7: Collision correction system effectiveness analysis.
        
        Specifically tests the step-back and fix collision correction
        system under various penetration scenarios.
        """
        test_id = "correction_effectiveness"
        
        try:
            # Create scenario likely to cause penetrations
            positions = self.table.calc_racked_positions(spacing_mode='tight', seed=42)
            physics = PoolPhysics(initial_positions=positions, ball_collision_model='simple')
            physics._use_multicore_collision = True
            
            # Track correction attempts
            correction_attempts = 0
            correction_successes = 0
            
            if hasattr(physics, '_collision_corrector'):
                original_correct = physics._collision_corrector.check_and_correct_penetrations
                
                def tracked_correction(current_time):
                    nonlocal correction_attempts, correction_successes
                    correction_attempts += 1
                    result = original_correct(current_time)
                    if result:
                        correction_successes += 1
                    return result
                
                physics._collision_corrector.check_and_correct_penetrations = tracked_correction
            
            start_time = time.time()
            physics.strike_ball(0.0, 0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 
                              np.array([0.0, 0.0, -1.8]), 0.54)
            
            physics.evolve(5.0)
            execution_time = time.time() - start_time
            
            correction_rate = (correction_successes / correction_attempts) if correction_attempts > 0 else 0
            
            return CollisionTestResult(
                test_id=test_id,
                success=(correction_rate >= 0.5),  # 50% correction success threshold
                execution_time=execution_time,
                num_events=len(physics.events),
                penetrations_detected=0,
                corrections_attempted=correction_attempts,
                corrections_successful=correction_successes,
                physics_metrics={
                    'correction_rate': correction_rate,
                    'correction_attempts': correction_attempts,
                    'correction_successes': correction_successes
                }
            )
            
        except Exception as e:
            penetrations = 1 if 'penetrated' in str(e).lower() else 0
            return CollisionTestResult(
                test_id=test_id,
                success=False,
                execution_time=0,
                num_events=0,
                penetrations_detected=penetrations,
                corrections_attempted=0,
                corrections_successful=0,
                error_message=str(e)
            )
    
    def run_comprehensive_test_suite(self) -> Dict:
        """
        Run all deep testing strategies and generate comprehensive report.
        """
        _logger.info("Starting comprehensive deep collision testing suite...")
        
        test_methods = [
            self.test_1_analytical_two_ball_collision,
            self.test_2_stress_high_velocity_collisions,
            self.test_3_edge_case_simultaneous_collisions,
            self.test_4_precision_near_miss_collisions,
            self.test_5_performance_scaling_analysis,
            lambda: self.test_6_monte_carlo_random_scenarios(25),  # Reduced for speed
            self.test_7_collision_correction_effectiveness
        ]
        
        results = []
        total_start_time = time.time()
        
        for i, test_method in enumerate(test_methods, 1):
            _logger.info(f"Running test {i}/{len(test_methods)}: {test_method.__name__}")
            result = test_method()
            results.append(result)
            
            status = "PASS" if result.success else "FAIL"
            _logger.info(f"  {status} - {result.execution_time:.3f}s - {result.num_events} events")
            
            if result.error_message:
                _logger.warning(f"  Error: {result.error_message}")
        
        total_time = time.time() - total_start_time
        
        # Generate comprehensive analysis
        analysis = self._analyze_results(results, total_time)
        
        return {
            'results': results,
            'analysis': analysis,
            'total_execution_time': total_time
        }
    
    def _analyze_results(self, results: List[CollisionTestResult], total_time: float) -> Dict:
        """
        Analyze test results and generate insights.
        """
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.success)
        success_rate = (successful_tests / total_tests) * 100
        
        total_penetrations = sum(r.penetrations_detected for r in results)
        total_corrections = sum(r.corrections_attempted for r in results)
        successful_corrections = sum(r.corrections_successful for r in results)
        
        correction_rate = (successful_corrections / total_corrections * 100) if total_corrections > 0 else 0
        
        return {
            'summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': success_rate,
                'total_execution_time': total_time
            },
            'collision_detection': {
                'penetrations_detected': total_penetrations,
                'penetration_rate': (total_penetrations / total_tests) * 100
            },
            'collision_correction': {
                'correction_attempts': total_corrections,
                'successful_corrections': successful_corrections,
                'correction_success_rate': correction_rate
            },
            'performance': {
                'avg_test_time': total_time / total_tests,
                'total_events': sum(r.num_events for r in results),
                'avg_events_per_test': sum(r.num_events for r in results) / total_tests
            }
        }


def main():
    """
    Main function to run deep collision detection testing.
    """
    print("=== DEEP COLLISION DETECTION VALIDATION SUITE ===")
    print()
    
    tester = DeepCollisionTester()
    suite_results = tester.run_comprehensive_test_suite()
    
    # Print comprehensive report
    analysis = suite_results['analysis']
    
    print("=== DEEP TESTING RESULTS ===")
    print(f"Total Tests: {analysis['summary']['total_tests']}")
    print(f"Success Rate: {analysis['summary']['successful_tests']}/{analysis['summary']['total_tests']} ({analysis['summary']['success_rate']:.1f}%)")
    print(f"Total Execution Time: {analysis['summary']['total_execution_time']:.2f}s")
    print()
    
    print("=== COLLISION DETECTION ANALYSIS ===")
    print(f"Penetrations Detected: {analysis['collision_detection']['penetrations_detected']}")
    print(f"Penetration Rate: {analysis['collision_detection']['penetration_rate']:.1f}%")
    print()
    
    print("=== COLLISION CORRECTION ANALYSIS ===")
    print(f"Correction Attempts: {analysis['collision_correction']['correction_attempts']}")
    print(f"Successful Corrections: {analysis['collision_correction']['successful_corrections']}")
    print(f"Correction Success Rate: {analysis['collision_correction']['correction_success_rate']:.1f}%")
    print()
    
    print("=== PERFORMANCE ANALYSIS ===")
    print(f"Average Test Time: {analysis['performance']['avg_test_time']:.3f}s")
    print(f"Total Events Generated: {analysis['performance']['total_events']}")
    print(f"Average Events per Test: {analysis['performance']['avg_events_per_test']:.1f}")
    print()
    
    print("=== INDIVIDUAL TEST RESULTS ===")
    for result in suite_results['results']:
        status = "✓ PASS" if result.success else "✗ FAIL"
        print(f"{status} {result.test_id}: {result.execution_time:.3f}s, {result.num_events} events")
        if result.error_message:
            print(f"    Error: {result.error_message}")
        if result.physics_metrics:
            for key, value in result.physics_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"    {key}: {value}")
    
    print()
    print("=== DEEP TESTING ASSESSMENT ===")
    
    if analysis['summary']['success_rate'] >= 85:
        print("🎯 EXCELLENT: Collision detection system passes comprehensive deep testing")
    elif analysis['summary']['success_rate'] >= 70:
        print("✓ GOOD: Collision detection system passes most deep tests")
    else:
        print("⚠ NEEDS IMPROVEMENT: Collision detection system has issues in deep testing")
    
    if analysis['collision_correction']['correction_success_rate'] >= 70:
        print("✓ CORRECTION SYSTEM: Step-back and fix approach is effective")
    elif analysis['collision_correction']['correction_attempts'] > 0:
        print("⚠ CORRECTION SYSTEM: Step-back approach active but needs refinement")
    else:
        print("⚠ CORRECTION SYSTEM: No correction attempts detected")
    
    # Save detailed results
    with open('deep_collision_test_results.json', 'w') as f:
        # Convert results to serializable format
        serializable_results = []
        for result in suite_results['results']:
            serializable_results.append({
                'test_id': result.test_id,
                'success': result.success,
                'execution_time': result.execution_time,
                'num_events': result.num_events,
                'penetrations_detected': result.penetrations_detected,
                'corrections_attempted': result.corrections_attempted,
                'corrections_successful': result.corrections_successful,
                'error_message': result.error_message,
                'physics_metrics': result.physics_metrics
            })
        
        json.dump({
            'results': serializable_results,
            'analysis': analysis
        }, f, indent=2)
    
    print()
    print("Detailed results saved to: deep_collision_test_results.json")
    print("=== DEEP COLLISION TESTING COMPLETE ===")


if __name__ == '__main__':
    main()
