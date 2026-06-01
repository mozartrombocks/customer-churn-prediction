import subprocess
import sys

scripts = [
    "src/data_prep.py",
    "src/train.py",
    "src/evaluate.py",
    "src/feature_importance.py"
    "src/eda.py"
]

for script in scripts:
    print(f"Running {scripts}...")
    result = subprocess.run([sys.executable, script])
    
    if result.returncode != 0:
        print(f"Error running {script}:")
        break
else: 
    print("Pipeline completed successfully.")

