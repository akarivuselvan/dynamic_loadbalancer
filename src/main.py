#!/usr/bin/env python3
"""
Main runner for dynamic load balancing simulation.
"""

"""
Main runner for dynamic load balancing simulation.
"""

from simulator import Simulator
from strategies import RoundRobinStrategy, RandomStrategy
from pso import PSOStrategy
from utils import save_results

import os

from src.utils import save_results


# Simulation parameters
NUM_VMS = 6
SIM_TIME = 1000          # total simulated time units
ARRIVAL_RATE = 0.5       # average tasks per time unit (Poisson)
BATCH_INTERVAL = 10      # PSO batch interval
SEED = 42

# Output directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "sample_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_all():
    strategies = {
        "PSO": PSOStrategy(num_vms=NUM_VMS,
                           batch_interval=BATCH_INTERVAL,
                           seed=SEED),
        "RoundRobin": RoundRobinStrategy(num_vms=NUM_VMS),
        "Random": RandomStrategy(num_vms=NUM_VMS, seed=SEED)
    }

    results = []
    for name, strat in strategies.items():
        print(f"\n=== Running strategy: {name} ===")
        sim = Simulator(
            num_vms=NUM_VMS,
            sim_time=SIM_TIME,
            arrival_rate=ARRIVAL_RATE,
            strategy=strat,
            batch_interval=BATCH_INTERVAL,
            seed=SEED
        )
        res = sim.run()
        res["strategy"] = name
        results.append(res)
        save_results(res, OUTPUT_DIR, name)

    # Aggregate summary
    import pandas as pd
    df = pd.DataFrame(results).set_index("strategy")
    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    df.to_csv(summary_path)
    print("\n=== Summary ===")
    print(df)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    run_all()
