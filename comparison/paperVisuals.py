
import pandas as pd
import matplotlib.pyplot as plt

# Data for e5-multilingual-instruct
data_e5 = [
    {'Benchmark':'AlloprofReranking','Subset':'default','Main Score':0.734721},
    {'Benchmark':'AlloprofRetrieval','Subset':'default','Main Score':0.52065},
    {'Benchmark':'BUCC.v2','Subset':'de-en','Main Score':0.995825},
    {'Benchmark':'BUCC.v2','Subset':'fr-en','Main Score':0.990627},
    {'Benchmark':'BelebeleRetrieval','Subset':'deu_Latn-eng_Latn','Main Score':0.94574},
    {'Benchmark':'BelebeleRetrieval','Subset':'eng_Latn-deu_Latn','Main Score':0.93021},
    {'Benchmark':'BelebeleRetrieval','Subset':'eng_Latn-fra_Latn','Main Score':0.93471},
    {'Benchmark':'BelebeleRetrieval','Subset':'fra_Latn-eng_Latn','Main Score':0.94728},
    {'Benchmark':'DiaBlaBitextMining','Subset':'en-fr','Main Score':0.863342},
    {'Benchmark':'DiaBlaBitextMining','Subset':'fr-en','Main Score':0.863342},
    {'Benchmark':'STS17','Subset':'en-de','Main Score':0.850932},
    {'Benchmark':'STS17','Subset':'en-en','Main Score':0.876358},
    {'Benchmark':'STS17','Subset':'es-en','Main Score':0.847173},
    {'Benchmark':'STS17','Subset':'es-es','Main Score':0.880871},
    {'Benchmark':'STS17','Subset':'fr-en','Main Score':0.832552},
    {'Benchmark':'STS17','Subset':'nl-en','Main Score':0.846465},
    {'Benchmark':'STSES','Subset':'default','Main Score':0.750102},
    {'Benchmark':'StatcanDialogueDatasetRetrieval','Subset':'english','Main Score':0.43348},
    {'Benchmark':'StatcanDialogueDatasetRetrieval','Subset':'french','Main Score':0.26929},
    {'Benchmark':'WikipediaRerankingMultilingual','Subset':'de','Main Score':0.886117},
    {'Benchmark':'WikipediaRerankingMultilingual','Subset':'en','Main Score':0.889223},
    {'Benchmark':'WikipediaRetrievalMultilingual','Subset':'de','Main Score':0.92544},
    {'Benchmark':'WikipediaRetrievalMultilingual','Subset':'en','Main Score':0.94141},
]

# Data for best-of ensemble
data_ensemble = [
    {'Benchmark':'AlloprofReranking','Subset':'default','Main Score':0.704598},
    {'Benchmark':'AlloprofRetrieval','Subset':'default','Main Score':0.41261},
    {'Benchmark':'BUCC.v2','Subset':'de-en','Main Score':0.931531},
    {'Benchmark':'BUCC.v2','Subset':'fr-en','Main Score':0.981675},
    {'Benchmark':'BelebeleRetrieval','Subset':'deu_Latn-eng_Latn','Main Score':0.93281},
    {'Benchmark':'BelebeleRetrieval','Subset':'eng_Latn-deu_Latn','Main Score':0.91109},
    {'Benchmark':'BelebeleRetrieval','Subset':'eng_Latn-fra_Latn','Main Score':0.92148},
    {'Benchmark':'BelebeleRetrieval','Subset':'fra_Latn-eng_Latn','Main Score':0.9346},
    {'Benchmark':'DiaBlaBitextMining','Subset':'en-fr','Main Score':0.820287},
    {'Benchmark':'DiaBlaBitextMining','Subset':'fr-en','Main Score':0.820287},
    {'Benchmark':'STS17','Subset':'en-de','Main Score':0.783898},
    {'Benchmark':'STS17','Subset':'en-en','Main Score':0.896612},
    {'Benchmark':'STS17','Subset':'es-en','Main Score':0.823394},
    {'Benchmark':'STS17','Subset':'es-es','Main Score':0.87355},
    {'Benchmark':'STS17','Subset':'fr-en','Main Score':0.820855},
    {'Benchmark':'STS17','Subset':'nl-en','Main Score':0.820113},
    {'Benchmark':'STSES','Subset':'default','Main Score':0.81362},
    {'Benchmark':'StatcanDialogueDatasetRetrieval','Subset':'english','Main Score':0.39135},
    {'Benchmark':'StatcanDialogueDatasetRetrieval','Subset':'french','Main Score':0.12963},
    {'Benchmark':'WikipediaRerankingMultilingual','Subset':'de','Main Score':0.863503},
    {'Benchmark':'WikipediaRerankingMultilingual','Subset':'en','Main Score':0.903638},
    {'Benchmark':'WikipediaRetrievalMultilingual','Subset':'de','Main Score':0.81037},
    {'Benchmark':'WikipediaRetrievalMultilingual','Subset':'en','Main Score':0.92434},
]

# Create DataFrames
df_e5 = pd.DataFrame(data_e5)
df_ensemble = pd.DataFrame(data_ensemble)

# Merge on Benchmark and Subset to compare
df_compare = pd.merge(
    df_e5, df_ensemble,
    on=['Benchmark','Subset'],
    how='inner',
    suffixes=('_e5','_ensemble')
)

# Compute difference
df_compare['Difference'] = df_compare['Main Score_e5'] - df_compare['Main Score_ensemble']

# Save a table of results
df_compare.to_csv('benchmark_results_comparison.csv', index=False)

# Plot the difference
plt.figure(figsize=(12,6))
plt.bar(df_compare['Benchmark'] + " (" + df_compare['Subset'] + ")", df_compare['Difference'])
plt.xticks(rotation=90)
plt.ylabel('Score Difference (e5 - Ensemble)')
plt.title('Difference in Main Scores by Benchmark and Subset')
plt.tight_layout()
plt.savefig('benchmark_score_differences.png')
plt.show()


