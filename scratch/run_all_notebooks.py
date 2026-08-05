"""
Scratch Script: Execute All Jupyter Notebooks and Save Cell Outputs
-------------------------------------------------------------------
Runs nbconvert ExecutePreprocessor on all 3 notebooks and saves outputs.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

# First build the clean notebooks
import scratch.build_and_execute_notebooks

notebooks = [
    project_root / "notebooks" / "01_exploratory_data_analysis.ipynb",
    project_root / "notebooks" / "02_model_training_and_eval.ipynb",
    project_root / "notebooks" / "ames_ds_pipeline.ipynb"
]

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

for nb_path in notebooks:
    print(f"[EXECUTING NOTEBOOK] {nb_path.name}...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)
        
    ep.preprocess(nb, {'metadata': {'path': str(nb_path.parent)}})
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    print(f" -> [PASSED] {nb_path.name} executed & outputs saved successfully!")

print("\n[ALL 3 JUPYTER NOTEBOOKS EXECUTED CLEANLY WITH OUTPUTS SAVED!]")
