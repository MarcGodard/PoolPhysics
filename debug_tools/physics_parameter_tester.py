#!/usr/bin/env python3
"""
Physics Parameter Testing Framework

Test and validate physics engine parameters against real-world pool physics.
Allows easy adjustment of friction, elasticity, and material properties.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pool_physics.table import PoolTable
from pool_physics import PoolPhysics
from pool_physics.events import BallSlidingEvent

class PhysicsParameterTester:
    """Framework for testing physics engine parameters"""
    
    # Default physics parameters (current engine values)
    DEFAULT_PARAMS = {
        # Ball properties
        'ball_mass': 0.1406,        # kg (5 oz regulation)
        'ball_radius': 0.02625,     # m (2.25 inch diameter)
        
        # Friction coefficients
        'mu_s': 0.21,               # sliding friction (ball-table)
        'mu_b': 0.05,               # ball-to-ball collision friction
        'mu_r': 0.016,              # rolling friction (ball-table)
        'mu_sp': 0.044,             # spinning friction (ball-table)
        
        # Elasticity/bounce
        'e': 0.89,                  # ball-to-ball restitution coefficient
        'kappa_rail': 0.6,          # rail restitution coefficient
        'kappa_corner': 0.6,        # corner restitution coefficient
        
        # Environment
        'g': 9.81                   # gravity (m/s²)
    }
    
    def __init__(self):
        self.test_results = []
        
    def test_parameter_set(self, params, test_name="Custom", verbose=True):
        """Test a specific parameter set"""
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"TESTING: {test_name}")
            print(f"{'='*50}")
            
        # Initialize physics with custom parameters
        physics = PoolPhysics(
            ball_mass=params['ball_mass'],
            ball_radius=params['ball_radius'],
            mu_s=params['mu_s'],
            mu_b=params['mu_b'],
            mu_r=params['mu_r'],
            mu_sp=params['mu_sp'],
            e=params['e'],
            g=params['g']
        )
        
        # Set rail elasticity (these are class variables)
        from pool_physics.events import SegmentCollisionEvent, CornerCollisionEvent
        SegmentCollisionEvent.kappa = params['kappa_rail']
        CornerCollisionEvent.kappa = params['kappa_corner']
        
        table = PoolTable()
        
        # Run standard test scenarios
        results = {
            'params': params.copy(),
            'test_name': test_name,
            'scenarios': {}
        }
        
        # Test 1: Single ball roll distance
        results['scenarios']['roll_distance'] = self._test_roll_distance(physics, table, verbose)
        
        # Test 2: Rail bounce behavior
        results['scenarios']['rail_bounce'] = self._test_rail_bounce(physics, table, verbose)
        
        # Test 3: Ball-to-ball collision
        results['scenarios']['ball_collision'] = self._test_ball_collision(physics, table, verbose)
        
        # Test 4: Break shot power
        results['scenarios']['break_power'] = self._test_break_power(physics, table, verbose)
        
        self.test_results.append(results)
        return results
    
    def _test_roll_distance(self, physics, table, verbose=True):
        """Test how far a ball rolls with given friction"""
        
        # Set up single ball rolling test
        initial_pos = np.array([0.0, table.H, 0.0])
        initial_velocity = np.array([2.0, 0.0, 0.0])  # 2 m/s sideways
        
        physics.reset(balls_on_table=[0], ball_positions=np.array([initial_pos]))
        
        # Create rolling event
        roll_event = BallSlidingEvent(
            t=0.0, i=0, r_0=initial_pos, 
            v_0=initial_velocity, omega_0=np.array([0.0, 0.0, 0.0])
        )
        
        events = physics.add_event_sequence(roll_event)
        
        # Find when ball stops
        final_time = 5.0
        final_pos = physics.eval_positions(final_time)[0]
        roll_distance = abs(final_pos[0] - initial_pos[0])
        
        if verbose:
            print(f"Roll Distance Test:")
            print(f"  Initial velocity: {np.linalg.norm(initial_velocity):.2f} m/s")
            print(f"  Roll distance: {roll_distance:.3f} m")
            print(f"  Events generated: {len(events)}")
            
        return {
            'initial_speed': np.linalg.norm(initial_velocity),
            'roll_distance': roll_distance,
            'events': len(events)
        }
    
    def _test_rail_bounce(self, physics, table, verbose=True):
        """Test rail bounce elasticity in both X and Z directions"""
        results = {}
        
        # Test Z-direction bounce (toward +Z rail)
        initial_pos = np.array([0.0, table.H, 0.0])  # Center of table
        initial_velocity = np.array([0.0, 0.0, 3.0])  # 3 m/s toward +Z rail
        
        physics.reset(balls_on_table=[0], ball_positions=np.array([initial_pos]))
        
        roll_event = BallSlidingEvent(
            t=0.0, i=0, r_0=initial_pos,
            v_0=initial_velocity, omega_0=np.array([0.0, 0.0, 0.0])
        )
        
        events = physics.add_event_sequence(roll_event)
        rail_collisions_z = [e for e in events if 'Segment' in type(e).__name__]
        
        if rail_collisions_z:
            collision_time = rail_collisions_z[0].t
            pre_collision_vel = physics.eval_velocities(collision_time - 0.001)[0]
            post_collision_vel = physics.eval_velocities(collision_time + 0.001)[0]
            
            speed_before_z = abs(pre_collision_vel[2])  # Z-direction
            speed_after_z = abs(post_collision_vel[2])
            bounce_ratio_z = speed_after_z / speed_before_z if speed_before_z > 0 else 0
        else:
            bounce_ratio_z = speed_before_z = speed_after_z = 0
            
        results['z_direction'] = {
            'speed_before': speed_before_z,
            'speed_after': speed_after_z,
            'bounce_ratio': bounce_ratio_z,
            'rail_collisions': len(rail_collisions_z)
        }
        
        # Test X-direction bounce (toward +X rail)
        initial_pos = np.array([0.0, table.H, 0.0])  # Center of table
        initial_velocity = np.array([3.0, 0.0, 0.0])  # 3 m/s toward +X rail
        
        physics.reset(balls_on_table=[0], ball_positions=np.array([initial_pos]))
        
        roll_event = BallSlidingEvent(
            t=0.0, i=0, r_0=initial_pos,
            v_0=initial_velocity, omega_0=np.array([0.0, 0.0, 0.0])
        )
        
        events = physics.add_event_sequence(roll_event)
        rail_collisions_x = [e for e in events if 'Segment' in type(e).__name__]
        
        if rail_collisions_x:
            collision_time = rail_collisions_x[0].t
            pre_collision_vel = physics.eval_velocities(collision_time - 0.001)[0]
            post_collision_vel = physics.eval_velocities(collision_time + 0.001)[0]
            
            speed_before_x = abs(pre_collision_vel[0])  # X-direction
            speed_after_x = abs(post_collision_vel[0])
            bounce_ratio_x = speed_after_x / speed_before_x if speed_before_x > 0 else 0
        else:
            bounce_ratio_x = speed_before_x = speed_after_x = 0
            
        results['x_direction'] = {
            'speed_before': speed_before_x,
            'speed_after': speed_after_x,
            'bounce_ratio': bounce_ratio_x,
            'rail_collisions': len(rail_collisions_x)
        }
        
        if verbose:
            print(f"Rail Bounce Test:")
            print(f"  Z-direction: {speed_before_z:.2f} → {speed_after_z:.2f} m/s (ratio: {bounce_ratio_z:.3f}, collisions: {len(rail_collisions_z)})")
            print(f"  X-direction: {speed_before_x:.2f} → {speed_after_x:.2f} m/s (ratio: {bounce_ratio_x:.3f}, collisions: {len(rail_collisions_x)})")
            
        return results
    
    def _test_ball_collision(self, physics, table, verbose=True):
        """Test ball-to-ball collision behavior"""
        
        # Set up two balls for head-on collision
        ball_spacing = 0.1  # 10cm apart
        pos1 = np.array([-ball_spacing/2, table.H, 0.0])
        pos2 = np.array([ball_spacing/2, table.H, 0.0])
        
        physics.reset(balls_on_table=[0, 1], ball_positions=np.array([pos1, pos2]))
        
        # Ball 0 moves toward ball 1
        initial_velocity = np.array([2.0, 0.0, 0.0])
        
        collision_event = BallSlidingEvent(
            t=0.0, i=0, r_0=pos1,
            v_0=initial_velocity, omega_0=np.array([0.0, 0.0, 0.0])
        )
        
        events = physics.add_event_sequence(collision_event)
        
        # Find ball collision
        ball_collisions = [e for e in events if 'BallCollision' in type(e).__name__]
        
        if ball_collisions:
            collision_time = ball_collisions[0].t
            post_velocities = physics.eval_velocities(collision_time + 0.001)
            
            if len(post_velocities) >= 2:
                v0_after = np.linalg.norm(post_velocities[0])
                v1_after = np.linalg.norm(post_velocities[1])
                energy_ratio = (v0_after**2 + v1_after**2) / np.linalg.norm(initial_velocity)**2
            else:
                v0_after = v1_after = energy_ratio = 0
        else:
            v0_after = v1_after = energy_ratio = 0
            
        if verbose:
            print(f"Ball Collision Test:")
            print(f"  Initial speed: {np.linalg.norm(initial_velocity):.2f} m/s")
            print(f"  Ball 0 after: {v0_after:.2f} m/s")
            print(f"  Ball 1 after: {v1_after:.2f} m/s")
            print(f"  Energy ratio: {energy_ratio:.3f}")
            print(f"  Ball collisions: {len(ball_collisions)}")
            
        return {
            'initial_speed': np.linalg.norm(initial_velocity),
            'ball0_after': v0_after,
            'ball1_after': v1_after,
            'energy_ratio': energy_ratio,
            'ball_collisions': len(ball_collisions)
        }
    
    def _test_break_power(self, physics, table, verbose=True):
        """Test break shot effectiveness"""
        
        # Standard 8-ball break
        initial_positions = table.calc_racked_positions(rack_type='8-ball')
        physics.reset(balls_on_table=list(range(16)), ball_positions=initial_positions)
        
        # Break shot
        cue_pos = initial_positions[0]
        target_pos = initial_positions[1]
        direction = target_pos - cue_pos
        direction[1] = 0
        direction_unit = direction / np.linalg.norm(direction)
        
        break_velocity = direction_unit * 12.0  # 12 m/s break
        
        break_event = BallSlidingEvent(
            t=0.0, i=0, r_0=cue_pos,
            v_0=break_velocity, omega_0=np.array([0.0, -5.0, 0.0])
        )
        
        events = physics.add_event_sequence(break_event)
        
        # Count collision types
        ball_collisions = len([e for e in events if 'BallCollision' in type(e).__name__])
        rail_collisions = len([e for e in events if 'Segment' in type(e).__name__])
        
        # Check final spread
        final_positions = physics.eval_positions(5.0)
        spreads = []
        for pos in final_positions[1:]:  # Skip cue ball
            if pos[1] > table.H - table.ball_radius:  # Still on table
                distance_from_center = np.sqrt(pos[0]**2 + pos[2]**2)
                spreads.append(distance_from_center)
        
        avg_spread = np.mean(spreads) if spreads else 0
        
        if verbose:
            print(f"Break Power Test:")
            print(f"  Break speed: {np.linalg.norm(break_velocity):.1f} m/s")
            print(f"  Ball collisions: {ball_collisions}")
            print(f"  Rail collisions: {rail_collisions}")
            print(f"  Average spread: {avg_spread:.3f} m")
            print(f"  Total events: {len(events)}")
            
        return {
            'break_speed': np.linalg.norm(break_velocity),
            'ball_collisions': ball_collisions,
            'rail_collisions': rail_collisions,
            'avg_spread': avg_spread,
            'total_events': len(events)
        }
    
    def compare_parameter_sets(self, param_sets, names=None):
        """Compare multiple parameter sets"""
        
        if names is None:
            names = [f"Set {i+1}" for i in range(len(param_sets))]
            
        print(f"\n{'='*60}")
        print("PARAMETER COMPARISON")
        print(f"{'='*60}")
        
        results = []
        for i, (params, name) in enumerate(zip(param_sets, names)):
            print(f"\nTesting {name}...")
            result = self.test_parameter_set(params, name, verbose=False)
            results.append(result)
            
        # Summary comparison
        print(f"\n{'COMPARISON SUMMARY':^60}")
        print("-" * 60)
        print(f"{'Test':<20} {'Metric':<15} {' | '.join(names)}")
        print("-" * 60)
        
        # Roll distance comparison
        distances = [r['scenarios']['roll_distance']['roll_distance'] for r in results]
        print(f"{'Roll Distance':<20} {'Distance (m)':<15} {' | '.join(f'{d:.3f}' for d in distances)}")
        
        # Rail bounce comparison (Z-direction)
        bounces_z = [r['scenarios']['rail_bounce']['z_direction']['bounce_ratio'] for r in results]
        print(f"{'Rail Bounce (Z)':<20} {'Ratio':<15} {' | '.join(f'{b:.3f}' for b in bounces_z)}")
        
        # Rail bounce comparison (X-direction)
        bounces_x = [r['scenarios']['rail_bounce']['x_direction']['bounce_ratio'] for r in results]
        print(f"{'Rail Bounce (X)':<20} {'Ratio':<15} {' | '.join(f'{b:.3f}' for b in bounces_x)}")
        
        # Ball collision comparison
        energies = [r['scenarios']['ball_collision']['energy_ratio'] for r in results]
        print(f"{'Ball Collision':<20} {'Energy Ratio':<15} {' | '.join(f'{e:.3f}' for e in energies)}")
        
        # Break power comparison
        spreads = [r['scenarios']['break_power']['avg_spread'] for r in results]
        print(f"{'Break Spread':<20} {'Avg Spread (m)':<15} {' | '.join(f'{s:.3f}' for s in spreads)}")
        
        return results
    
    def create_realistic_variants(self):
        """Create parameter sets for different table conditions"""
        
        variants = {}
        
        # Standard tournament conditions
        variants['Tournament'] = self.DEFAULT_PARAMS.copy()
        
        # Fast table (low friction, bouncy rails)
        variants['Fast Table'] = self.DEFAULT_PARAMS.copy()
        variants['Fast Table'].update({
            'mu_r': 0.012,      # Less rolling friction
            'mu_s': 0.18,       # Less sliding friction
            'kappa_rail': 0.75, # Bouncier rails
            'e': 0.92           # More elastic ball collisions
        })
        
        # Slow table (high friction, dead rails)
        variants['Slow Table'] = self.DEFAULT_PARAMS.copy()
        variants['Slow Table'].update({
            'mu_r': 0.022,      # More rolling friction
            'mu_s': 0.25,       # More sliding friction
            'kappa_rail': 0.45, # Deader rails
            'e': 0.85           # Less elastic collisions
        })
        
        # Worn table (inconsistent conditions)
        variants['Worn Table'] = self.DEFAULT_PARAMS.copy()
        variants['Worn Table'].update({
            'mu_r': 0.020,      # Higher friction
            'mu_sp': 0.055,     # More spin friction
            'kappa_rail': 0.55, # Inconsistent rails
            'e': 0.87           # Slightly less elastic
        })
        
        # New table (pristine conditions)
        variants['New Table'] = self.DEFAULT_PARAMS.copy()
        variants['New Table'].update({
            'mu_r': 0.014,      # Lower friction
            'mu_s': 0.19,       # Smoother sliding
            'kappa_rail': 0.70, # Consistent bouncy rails
            'e': 0.91           # More elastic
        })
        
        return variants

def demonstrate_parameter_testing():
    """Demonstrate the parameter testing framework"""
    
    tester = PhysicsParameterTester()
    
    print("PHYSICS PARAMETER TESTING FRAMEWORK")
    print("=" * 60)
    print("This framework allows testing different table conditions:")
    print("• Felt friction (mu_r, mu_s, mu_sp)")
    print("• Rail elasticity (kappa_rail, kappa_corner)")
    print("• Ball properties (mass, radius, restitution)")
    print("• Environmental factors (gravity)")
    
    # Test default parameters
    print(f"\n1. Testing default parameters...")
    default_result = tester.test_parameter_set(tester.DEFAULT_PARAMS, "Default Engine")
    
    # Test realistic variants
    print(f"\n2. Testing realistic table variants...")
    variants = tester.create_realistic_variants()
    
    # Compare a few variants
    test_variants = ['Tournament', 'Fast Table', 'Slow Table']
    param_sets = [variants[name] for name in test_variants]
    
    comparison_results = tester.compare_parameter_sets(param_sets, test_variants)
    
    print(f"\n{'='*60}")
    print("VALIDATION SUGGESTIONS:")
    print("=" * 60)
    print("To validate against real physics:")
    print("1. Measure actual roll distances on different tables")
    print("2. Test rail bounce ratios with consistent shots")
    print("3. Record break shot spread patterns")
    print("4. Compare collision energy transfer")
    print("5. Adjust parameters to match real measurements")
    
    return tester, comparison_results

if __name__ == "__main__":
    demonstrate_parameter_testing()
