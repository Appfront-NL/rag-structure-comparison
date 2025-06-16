# Python script to save benchmark tables as image files without inline display

import pandas as pd
import matplotlib.pyplot as plt

# Prepare data without subset column (subset details in label)
e5_data = [
    ["AlloprofReranking", 0.734721],
    ["AlloprofRetrieval", 0.52065],
    ["BUCC.v2 (de-en)", 0.995825],
    ["BUCC.v2 (fr-en)", 0.990627],
    ["BelebeleRetrieval (deu-eng)", 0.94574],
    ["BelebeleRetrieval (eng-deu)", 0.93021],
    ["BelebeleRetrieval (eng-fra)", 0.93471],
    ["BelebeleRetrieval (fra-eng)", 0.94728],
    ["DiaBlaBitextMining (en-fr)", 0.863342],
    ["DiaBlaBitextMining (fr-en)", 0.863342],
    ["STS12", 0.822449],
    ["STS17 (en-de)", 0.850932],
    ["STS17 (en-en)", 0.876358],
    ["STS17 (es-en)", 0.847173],
    ["STS17 (es-es)", 0.880871],
    ["STS17 (fr-en)", 0.832552],
    ["STS17 (nl-en)", 0.846465],
    ["STSES", 0.750102],
    ["StatcanDialogueDatasetRetrieval (en)", 0.43348],
    ["StatcanDialogueDatasetRetrieval (fr)", 0.26929],
    ["WikipediaRerankingMultilingual (de)", 0.886117],
    ["WikipediaRerankingMultilingual (en)", 0.889223],
    ["WikipediaRetrievalMultilingual (de)", 0.92544],
    ["WikipediaRetrievalMultilingual (en)", 0.94141],
]

ensemble_data = [
    ["AlloprofReranking", 0.704598],
    ["AlloprofRetrieval", 0.41261],
    ["BUCC.v2 (de-en)", 0.931531],
    ["BUCC.v2 (fr-en)", 0.981675],
    ["BelebeleRetrieval (deu-eng)", 0.93281],
    ["BelebeleRetrieval (eng-deu)", 0.91109],
    ["BelebeleRetrieval (eng-fra)", 0.92148],
    ["BelebeleRetrieval (fra-eng)", 0.9346],
    ["DiaBlaBitextMining (en-fr)", 0.820287],
    ["DiaBlaBitextMining (fr-en)", 0.820287],
    ["STS17 (en-de)", 0.783898],
    ["STS17 (en-en)", 0.896612],
    ["STS17 (es-en)", 0.823394],
    ["STS17 (es-es)", 0.87355],
    ["STS17 (fr-en)", 0.820855],
    ["STS17 (nl-en)", 0.820113],
    ["STSES", 0.81362],
    ["StatcanDialogueDatasetRetrieval (en)", 0.39135],
    ["StatcanDialogueDatasetRetrieval (fr)", 0.12963],
    ["WikipediaRerankingMultilingual (de)", 0.863503],
    ["WikipediaRerankingMultilingual (en)", 0.903638],
    ["WikipediaRetrievalMultilingual (de)", 0.81037],
    ["WikipediaRetrievalMultilingual (en)", 0.92434],
]

# Convert to DataFrames
e5_df = pd.DataFrame(e5_data, columns=["Benchmark", "Score"])
ensemble_df = pd.DataFrame(ensemble_data, columns=["Benchmark", "Score"])

# Function to create and save table image
def create_table_image(df, filename, title):
    fig, ax = plt.subplots(figsize=(8, len(df)*0.3 + 1))
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)
    plt.title(title, fontsize=12, pad=20)
    plt.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)

# Save table images to files
create_table_image(e5_df, 'e5_table_nosubset.png', 'E5-Multilingual-Instruct Scores')
create_table_image(ensemble_df, 'ensemble_table_nosubset.png', 'Ensemble Model Scores')

print("Saved e5_table_nosubset.png and ensemble_table_nosubset.png")
