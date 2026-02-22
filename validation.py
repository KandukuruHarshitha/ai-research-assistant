"""
validation.py
-------------
Master validation script to run All Unit Tests and Benchmarks.
"""

import subprocess
import sys
import os

def run_step(name, command):
    print(f"\n{'='*50}")
    print(f"RUNNING: {name}")
    print(f"{'='*50}")
    try:
        # result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        # Using simple run to see output live
        subprocess.run(command, shell=True, check=True)
        print(f"\n✅ {name} PASSED")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {name} FAILED")
        # print(e.output)

if __name__ == "__main__":
    print("🧪 Starting Full System Validation...")
    
    # 1. Run Unit Tests
    run_step("Unit Tests (PyTest)", f"{sys.executable} -m pytest tests/")
    
    # 2. Run Benchmarks
    run_step("Quantitative Benchmarks", f"{sys.executable} benchmark.py")
    
    print(f"\n{'='*50}")
    print("🏁 Validation Complete")
    print(f"{'='*50}")
