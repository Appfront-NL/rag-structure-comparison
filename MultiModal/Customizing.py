import mteb
from sentence_transformers import SentenceTransformer

# Step 1: Load the model using SentenceTransformers     SentenceTransformer("thenlper/gte-small"),

# model_name = "Lajavaness/bilingual-embedding-base"
model_name = "thenlper/gte-small"
model = SentenceTransformer(model_name, trust_remote_code=True)

# Step 2: Load the task (AlloprofRetrieval)
tasks = mteb.get_tasks(tasks=["AlloprofRetrieval"])

# Step 3: Create the evaluation pipeline
evaluation = mteb.MTEB(tasks=tasks)

# Step 4: Run the evaluation and save results to a specific folder
results = evaluation.run(
    model,
    output_folder="MultiModal/single/gte-small",
)

# (Optional) print the results
print("Evaluation completed. Results saved to:", "MultiModal/single/gte-small")
