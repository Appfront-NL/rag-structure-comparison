# RAG Structure Comparison

This project benchmarks and compares Retrieval-Augmented Generation (RAG) structures across diverse evaluation setups. It is designed to run large-scale, reproducible experiments using `sbatch` on a compute cluster. The focus is currently on multilingual and (soon) multimodal tasks using robust embedding models.

---

## 📂 Folder Structure

### `MultiLingual/`

This folder contains all experiments related to multilingual evaluations using the MTEB (Massive Text Embedding Benchmark) Europe benchmark.

**Subfolders and contents:**

- `template/`: Workspace for job generation and templates
  - `gen_jobs.py`: Script to auto-generate one Python and one Slurm job script per model × task
  - `PY_TEMPLATE.py`: Template used to generate Python runners
  - `SH_TEMPLATE.sh`: Template used to generate Slurm `.sh` jobs
  - `python_tasks/`: Auto-generated Python scripts, one for each task and model
  - `shell_jobs/`: Auto-generated Slurm `.sh` scripts to run each experiment
  - `logs/`: Output and error logs for each job (named by job ID)
  - `results/`: Generated `.csv` files with evaluation metrics for each run

### `multiModal/`

The `multiModal/` folder supports evaluation of **text-only embedding models** in a multimodal context, using the same task structure as the MTEB Europe benchmark. While originally designed to accommodate multimodal models, the current implementation focuses on evaluating purely textual models across tasks that may span different domains or data sources.

**Key contents:**

- `multiModalRun.py`: The main script for evaluating a set of predefined models across selected tasks.
- `ensambleTest.py`: Optional/experimental script for testing ensemble approaches.
- `LATE-Large-MM.sh`: Example Slurm script for scheduling a batch run of multimodal evaluations.
- `ResultsMultiModal/`: Directory where evaluation results are saved per model.

#### 📌 Running `multiModalRun.py`

To evaluate all models listed in the script, navigate to the `multiModal/` directory and run:

```bash
python multiModalRun.py
```
