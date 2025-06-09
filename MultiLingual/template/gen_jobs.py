import os

# List of models
models = [
    # "intfloat/multilingual-e5-large-instruct",
    # "HIT-TMG/KaLM-embedding-multilingual-mini-v1",
    # "BAAI/bge-m3"
    "aari1995/German_Semantic_V3b"
]

# List of tasks
tasks = [
    "AlloprofRetrieval",
    "StatcanDialogueDatasetRetrieval",
    "WikipediaRetrievalMultilingual",
    "BelebeleRetrieval",
    "AlloprofReranking",
    "WikipediaRerankingMultilingual",
    "DiaBLaBitextMining",
    "BUCCBitextMiningFast",
    "STS17Crosslingual",
    "STSES",
]

# Ensure folders exist
os.makedirs("python_tasks", exist_ok=True)
os.makedirs("shell_jobs", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)

def alias(model_name):
    return model_name.replace("/", "_").replace("-", "_")

# Load templates
with open("PY_TEMPLATE.py") as f:
    py_template = f.read()

with open("SH_TEMPLATE.sh") as f:
    sh_template = f.read()

# Generate files
for model in models:
    for task in tasks:
        a = alias(model)
        py_path = f"python_tasks/{task}___{a}.py"
        sh_path = f"shell_jobs/{task}___{a}.sh"
        with open(py_path, "w") as f:
            f.write(py_template.format(model=model, task=task))
        with open(sh_path, "w") as f:
            f.write(sh_template.format(task=task, alias=a))
