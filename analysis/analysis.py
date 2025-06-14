import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(path):
    df = pd.read_csv(path, sep=';', header=None)
    df.columns = [
        "Split", "NDCG@10", "NDCG@30", "Recall@10", "Recall@30",
        "DP@10", "DP@30", "EO@10", "EO@30", "F1@10", "F1@30",
        "Precision@10", "Precision@30", "MRR@10", "MRR@30"
    ]
    for col in df.columns[1:]:
        df[col] = df[col].astype(str).str.replace(",", ".").astype(float)
    return df


def analysis(base_file, new_file):
    base_results = load_data(base_file)
    new_results = load_data(new_file)

    diff_df = new_results.copy()
    for col in diff_df.columns[1:]:
        diff_df[col] =  base_results[col] - new_results[col]

    diff_df.set_index("Split", inplace=True)
    diff_df.to_csv(f"differences.csv")

    plt.figure(figsize=(14, 7))
    sns.heatmap(diff_df, annot=True, cmap="RdBu_r", center=0, fmt=".4f")
    plt.title(f"Różnice w wartościach metryk dla FairIB w stosunku do baseline")
    plt.ylabel("Próg podziału względem wieku")
    plt.xlabel("Metryki")
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"results.png")


analysis("fair.csv", "baselines.csv")