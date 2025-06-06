# Sourceduty Optimization Engine v1 ~ Drone Paths

"""
Applying a Sourceduty Optimization Engine to a real-world problem: optimizing the path of an autonomous drone.
It leverages the Navisol, TolSum, OptRef, and ImpactQ frameworks to minimize energy usage (as a function of distance),
enforce spatial constraints like obstacle avoidance and segment length, and refine feasible waypoints.
The engine computes optimal control points between a fixed start and end position, ensuring safe and efficient drone flight.
"""

import numpy as np
from scipy.optimize import minimize, Bounds

# --- Framework Modules ---
def apply_truthvar_logical_filter(constraints, bounds):
    return (bounds.lb + bounds.ub) / 2  

def apply_tolsum(x, constraints, tol):
    max_iter = 100
    for c in constraints:
        iter_count = 0
        while abs(c(x)) > tol and iter_count < max_iter:
            x -= 0.01 * np.sign(c(x))
            iter_count += 1
    return x

def apply_optref(x, bounds):
    return np.clip(x, bounds.lb, bounds.ub)

def compute_impactq(objective_fn, x):
    h = 1e-5
    impact = {}
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = h
        impact_value = abs((objective_fn(x + dx) - objective_fn(x)) / h)
        impact[f'x{i}'] = impact_value
    return impact

# --- Objective Function (Energy Minimization) ---
def energy_objective(x):
    P0 = np.array([0, 0])
    P1 = np.array([x[0], x[1]])
    P2 = np.array([x[2], x[3]])
    P3 = np.array([x[4], x[5]])
    P4 = np.array([5, 5])
    points = [P0, P1, P2, P3, P4]
    return sum(np.sum((points[i+1] - points[i])**2) for i in range(4))

# --- Constraints ---
def avoid_obstacle(x):
    min_dist = float('inf')
    for i in range(0, len(x), 2):
        dx = x[i] - 2
        dy = x[i+1] - 2
        dist_sq = dx**2 + dy**2
        min_dist = min(min_dist, dist_sq)
    return min_dist - 1.0

def segment_speed_constraint(x):
    points = [np.array([0, 0])] + [np.array([x[i], x[i+1]]) for i in range(0, len(x), 2)] + [np.array([5, 5])]
    max_dist = max(np.linalg.norm(points[i+1] - points[i]) for i in range(len(points)-1))
    return 3.0 - max_dist  # must be >= 0

# --- Optimization Engine ---
def Engine(objective_fn, constraints, bounds, tol=1e-3):
    x0 = apply_truthvar_logical_filter(constraints, bounds)
    cons = [{'type': 'ineq', 'fun': c} for c in constraints if callable(c)]
    result = minimize(objective_fn, x0, method='SLSQP', bounds=bounds, constraints=cons, tol=tol)
    
    if not result.success:
        print(f"Warning: Optimization failed - {result.message}")
    
    optimal_x = result.x if result.success else x0
    # Optionally refine (commented to avoid slowdown)
    # refined_x = apply_tolsum(optimal_x.copy(), constraints, tol)
    refined_x = optimal_x.copy()
    final_x = apply_optref(refined_x, bounds)
    impact = compute_impactq(objective_fn, final_x)

    return {
        'optimal_values': final_x,
        'impact': impact,
        'success': result.success,
        'message': result.message
    }

# --- Run the Engine ---
def test_drone_path_engine():
    bounds = Bounds([0, 0]*3, [5, 5]*3)
    constraints = [avoid_obstacle, segment_speed_constraint]
    result = Engine(energy_objective, constraints, bounds)

    print("\n=== Drone Path Optimization Result ===")
    print("Waypoints (x1, y1, x2, y2, x3, y3):")
    print(result['optimal_values'].reshape(3, 2))

    print("\nImpactQ Scores (Sensitivity):")
    for k, v in result['impact'].items():
        print(f"  {k}: {v:.4f}")

    print(f"\nSuccess: {result['success']}")
    print(f"Message: {result['message']}")
    return result

test_drone_path_engine()
