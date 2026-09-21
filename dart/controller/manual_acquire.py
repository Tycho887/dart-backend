import math
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class SatFinderNaive(ABC):
    def __init__(self, dither_amplitude: float, dither_period: float):
        self.offset = 0.0          
        self.A = dither_amplitude
        self.p = dither_period
        self.t = 0.0               
        self.locked = False

    def f(self, t: float, locked: bool) -> float:
        """The ODE: dx/dt = A·sin(2πt/p)·(1 − lock)."""
        return self.A * math.sin(2 * math.pi * t / self.p) * (not locked)

    def update(self, lock_state: bool, dt: float):
        """One Euler step: x ← x + f(t)·dt, then advance time."""
        self.locked = lock_state
        self.offset += self.f(self.t, lock_state) * dt
        self.t += dt

class SatFinderV2(ABC):
    def __init__(self, v_max: float, initial_period: float, growth_rate: float = 0.1):
        self.offset = 0.0          
        self.v_max = v_max         
        self.omega_0 = 2 * math.pi / initial_period
        self.c = growth_rate       
        self.t = 0.0               
        self.locked = False

    def f(self, t: float, locked: bool) -> float:
        """The ODE: dx/dt = V_max * sin( (omega_0/c) * ln(1 + c*t) ) * (1 - lock)."""
        if locked:
            return 0.0
            
        if self.c == 0.0:
            phase = self.omega_0 * t
        else:
            phase = (self.omega_0 / self.c) * math.log(1.0 + self.c * t)
            
        return self.v_max * math.sin(phase)

    def update(self, lock_state: bool, dt: float):
        """One Euler step: x <- x + f(t)*dt, then advance time."""
        self.locked = lock_state
        self.offset += self.f(self.t, lock_state) * dt
        self.t += dt

def run_monte_carlo(num_trials: int, dt: float, bound: float, std_dev: float):
    
    # Kinematic limits & tuning
    # Naive ODE uses velocity amplitude A
    A_naive, p_naive = 200.0, 600.0
    
    # V2 limits explicitly bound velocity while growing position envelope
    v_max, p_init, growth_rate = 20.0, 10.0, 0.05
    timeout = 900.0

    times_naive = []
    times_v2 = []

    np.random.seed(42)
    targets = np.random.normal(0.0, std_dev, num_trials)

    print(f"Running {num_trials} Monte Carlo trials...")
    
    for i, target in enumerate(targets):
        # print(f"target: {target}")
        # 1. Simulate Naive Strategy
        # finder_naive = SatFinderNaive(A_naive, p_naive)
        # while finder_naive.t < timeout:
        #     if abs(finder_naive.offset - target) < bound:
        #         break
            
        #     finder_naive.update(False, dt)
        # times_naive.append(finder_naive.t)

        # 2. Simulate V2 (Frequency Modulated) Strategy
        finder_v2 = SatFinderV2(v_max, p_init, growth_rate)
        while finder_v2.t < timeout:
            if abs(finder_v2.offset - target) < bound:
                break
            finder_v2.update(False, dt)
        times_v2.append(finder_v2.t)

        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{num_trials} trials")

    return times_naive, times_v2

def plot_distributions(times_v2):
    plt.figure(figsize=(12, 6))
    
    # Bin sizes are kept consistent across both distributions
    bins = np.histogram(times_v2, bins=30)[1]
    
    # plt.hist(times_naive, bins=bins, alpha=0.6, color='blue', label='SatFinderNaive (Constant)')
    plt.hist(times_v2, bins=bins, alpha=0.6, color='red', label='SatFinderV2 (Freq Modulated)')
    
    # Indicate means
    # plt.axvline(np.mean(times_naive), color='blue', linestyle='dashed', linewidth=1.5)
    plt.axvline(np.mean(times_v2), color='red', linestyle='dashed', linewidth=1.5)
    
    plt.title('Time-to-Acquire Distributions: Constant vs. Expanding Oscillations')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Trials)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    trials = 10000
    dt_step = 1.0
    tolerance_bound = 3.0
    gaussian_std = 20.0

    t_naive, t_v2 = run_monte_carlo(trials, dt_step, tolerance_bound, gaussian_std)
    
    print("\n--- Simulation Results ---")
    print(f"Naive Mean Time: {np.mean(t_naive):.1f} s")
    print(f"V2 Mean Time:    {np.mean(t_v2):.1f} s")
    
    plot_distributions(t_v2)