"""
Scratch Script: Pure Python Notebook Executor & Output Embedder
--------------------------------------------------------------
Executes all cells of Jupyter notebooks, capturing stdout, text outputs,
and Matplotlib/Seaborn plots as base64 images into the notebook outputs!
"""

import sys
import io
import base64
from pathlib import Path

sys.path = [p for p in sys.path if 'google-cloud-sdk' not in p]
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


import nbformat as nbf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# First ensure notebooks are built
import scratch.build_and_execute_notebooks

notebooks = [
    project_root / "notebooks" / "01_exploratory_data_analysis.ipynb",
    project_root / "notebooks" / "02_model_training_and_eval.ipynb",
    project_root / "notebooks" / "ames_ds_pipeline.ipynb"
]

for nb_path in notebooks:
    print(f"\n[EXECUTING CELL OUTPUTS FOR] {nb_path.name}...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)
        
    global_env = {'__name__': '__main__'}
    exec_count = 1
    
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = exec_count
            
            code = cell.source
            # Replace show() to capture figures
            plt.close('all')
            
            stdout_capture = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = stdout_capture
            
            try:
                # Handle path resolution inside notebook environment
                exec_code = code.replace("Path('../data/AmesHousing.csv')", "Path('data/AmesHousing.csv')")
                exec_code = exec_code.replace("Path('../')", "Path('.')")
                
                res = eval(compile(exec_code, f'<{nb_path.name}>', 'exec'), global_env)
                
                # Check captured stdout text
                stdout_text = stdout_capture.getvalue()
                if stdout_text:
                    cell.outputs.append(nbf.v4.new_output(
                        output_type='stream',
                        name='stdout',
                        text=stdout_text
                    ))
                    
                # Check for active matplotlib figures
                if plt.get_fignums():
                    for fig_num in plt.get_fignums():
                        fig = plt.figure(fig_num)
                        img_buf = io.BytesIO()
                        fig.savefig(img_buf, format='png', bbox_inches='tight')
                        img_buf.seek(0)
                        img_b64 = base64.b64encode(img_buf.read()).decode('utf-8')
                        
                        cell.outputs.append(nbf.v4.new_output(
                            output_type='display_data',
                            data={'image/png': img_b64, 'text/plain': '<Figure size ...>'},
                            metadata={}
                        ))
                    plt.close('all')
                    
            except Exception as e:
                stdout_text = stdout_capture.getvalue()
                print(f" -> [ERROR in cell {exec_count}]: {e}")
                cell.outputs.append(nbf.v4.new_output(
                    output_type='error',
                    ename=type(e).__name__,
                    evalue=str(e),
                    traceback=[str(e)]
                ))
            finally:
                sys.stdout = old_stdout
                
            exec_count += 1
            
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    print(f" -> [SUCCESS] Saved executed notebook with outputs: '{nb_path.name}'")

print("\n[ALL 3 JUPYTER NOTEBOOKS FULLY EXECUTED & VISUAL OUTPUTS EMBEDDED!] ")
