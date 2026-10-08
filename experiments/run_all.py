"""
Runs all three problems one after another.

    python run_all.py --quick     (fast check that everything works)
    python run_all.py             (full paper settings, takes a long time)
    python run_all.py --load      (re-make all figures from saved models)
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["consolidation.py", "heat.py", "spring_inverse.py"]

for script in SCRIPTS:
    print("\n" + "=" * 70)
    print(f"Running {script}")
    print("=" * 70)
    subprocess.run([sys.executable, os.path.join(ROOT, "experiments", script)]
                   + sys.argv[1:], check=True)

print("\nAll done. Figures and metrics are in the results/ folder.")
