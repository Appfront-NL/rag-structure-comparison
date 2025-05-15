import mteb
from sentence_transformers import SentenceTransformer

# Step 1: Load the model using SentenceTransformers
model_name = "ibm-granite/granite-embedding-107m-multilingual"
model = SentenceTransformer(model_name, trust_remote_code=True)

# Step 2: Load the task (AlloprofRetrieval)
tasks = mteb.get_tasks(tasks=["SwissJudgementClassification"],)

# Step 3: Create the evaluation pipeline
evaluation = mteb.MTEB(tasks=tasks)

# Step 4: Run the evaluation and save results to a specific folder
results = evaluation.run(
    model,
    output_folder="MultiModal/single/granite-embedding-107m-multilingual",
)

# (Optional) print the results
print("Evaluation completed. Results saved to:", "MultiModal/single/granite-embedding-107m-multilingual")
