import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# e5-multilingual-instruct model data
e5_data = [
    ["AlloprofReranking", "default", 0.734721],
    ["AlloprofRetrieval", "default", 0.52065],
    ["BUCC.v2", "de-en", 0.995825],
    ["BUCC.v2", "fr-en", 0.990627],
    ["BelebeleRetrieval", "deu_Latn-eng_Latn", 0.94574],
    ["BelebeleRetrieval", "eng_Latn-deu_Latn", 0.93021],
    ["BelebeleRetrieval", "eng_Latn-fra_Latn", 0.93471],
    ["BelebeleRetrieval", "fra_Latn-eng_Latn", 0.94728],
    ["DiaBlaBitextMining", "en-fr", 0.863342],
    ["DiaBlaBitextMining", "fr-en", 0.863342],
    ["STS12", "default", 0.822449],
    ["STS17", "en-de", 0.850932],
    ["STS17", "en-en", 0.876358],
    ["STS17", "es-en", 0.847173],
    ["STS17", "es-es", 0.880871],
    ["STS17", "fr-en", 0.832552],
    ["STS17", "nl-en", 0.846465],
    ["STSES", "default", 0.750102],
    ["StatcanDialogueDatasetRetrieval", "english", 0.43348],
    ["StatcanDialogueDatasetRetrieval", "french", 0.26929],
    ["WikipediaRerankingMultilingual", "de", 0.886117],
    ["WikipediaRerankingMultilingual", "en", 0.889223],
    ["WikipediaRetrievalMultilingual", "de", 0.92544],
    ["WikipediaRetrievalMultilingual", "en", 0.94141],
]

# Ensemble model data
ensemble_data = [
    ["AlloprofReranking", "default", 0.704598],
    ["AlloprofRetrieval", "default", 0.41261],
    ["BUCC.v2", "de-en", 0.931531],
    ["BUCC.v2", "fr-en", 0.981675],
    ["BelebeleRetrieval", "deu_Latn-eng_Latn", 0.93281],
    ["BelebeleRetrieval", "eng_Latn-deu_Latn", 0.91109],
    ["BelebeleRetrieval", "eng_Latn-fra_Latn", 0.92148],
    ["BelebeleRetrieval", "fra_Latn-eng_Latn", 0.9346],
    ["DiaBlaBitextMining", "en-fr", 0.820287],
    ["DiaBlaBitextMining", "fr-en", 0.820287],
    ["STS17", "en-de", 0.783898],
    ["STS17", "en-en", 0.896612],
    ["STS17", "es-en", 0.823394],
    ["STS17", "es-es", 0.87355],
    ["STS17", "fr-en", 0.820855],
    ["STS17", "nl-en", 0.820113],
    ["STSES", "default", 0.81362],
    ["StatcanDialogueDatasetRetrieval", "english", 0.39135],
    ["StatcanDialogueDatasetRetrieval", "french", 0.12963],
    ["WikipediaRerankingMultilingual", "de", 0.863503],
    ["WikipediaRerankingMultilingual", "en", 0.903638],
    ["WikipediaRetrievalMultilingual", "de", 0.81037],
    ["WikipediaRetrievalMultilingual", "en", 0.92434],
]

# Convert lists to DataFrames
e5_df = pd.DataFrame(e5_data, columns=["Benchmark", "Subset", "E5"])
ensemble_df = pd.DataFrame(ensemble_data, columns=["Benchmark", "Subset", "Ensemble"])

# Create a combined Task column for merging
e5_df["Task"] = e5_df["Benchmark"] + " | " + e5_df["Subset"]
ensemble_df["Task"] = ensemble_df["Benchmark"] + " | " + ensemble_df["Subset"]

# Merge on Task and compute difference
merged = pd.merge(
    e5_df[["Task", "E5"]],
    ensemble_df[["Task", "Ensemble"]],
    on="Task",
    how="inner"
)
merged["Difference"] = merged["E5"] - merged["Ensemble"]

# Extract just the Benchmark for labeling (remove language/subset notation)
merged["Benchmark"] = merged["Task"].apply(lambda x: x.split(" | ")[0])

# Sort by difference descending
merged = merged.sort_values("Difference", ascending=False).reset_index(drop=True)

# Save results
merged.to_csv("benchmark_results_comparison.csv", index=False)

# Plot difference bars with two colors
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(merged))
colors = ["tab:green" if val >= 0 else "tab:red" for val in merged["Difference"]]
ax.bar(x, merged["Difference"], color=colors)

# Add zero line
ax.axhline(0, color="black", linewidth=1)

# Labeling with only Benchmark names
ax.set_xticks(x)
ax.set_xticklabels(merged["Benchmark"], rotation=90, fontsize=8)
ax.set_ylabel("Score Difference (E5 - Ensemble)")
ax.set_title("E5-multilingual-instruct vs Ensemble: Difference per Task")
ax.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("benchmark_score_differences.png")
plt.show()
