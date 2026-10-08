"""
Runs all Part 2 improvements one after another.

    python run_part2.py --quick     (fast check that everything works)
    python run_part2.py             (full settings, same budget as Part 1)

Runs:
    1. experiments/heat_improved.py                     Fix 1: hard boundary condition
    2. experiments/consolidation_rar.py                 Fix 2: RAR adaptive points
    3. experiments/consolidation_rar.py --random-add    Control: same number of points, added at random

Run the Part 1 scripts first (python run_all.py), because the Part 2 scripts
compare against the saved Part 1 models in results/heat and results/consolidation.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = [
    ["heat_improved.py"],
    ["consolidation_rar.py"],
    ["consolidation_rar.py", "--random-add"],
]

for run in RUNS:
    print("\n" + "=" * 70)
    print("Running " + " ".join(run))
    print("=" * 70)
    cmd = [sys.executable, os.path.join(ROOT, "experiments", run[0])] + run[1:] + sys.argv[1:]
    subprocess.run(cmd, check=True)

print("\nAll Part 2 runs done. Results are in results/heat_improved, "
      "results/consolidation_rar and results/consolidation_rar_random_control.")
