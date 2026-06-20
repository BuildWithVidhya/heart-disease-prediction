import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dataset/heart.csv")

def generate_heatmap():

    plt.figure(figsize=(12,8))

    sns.heatmap(
        df.corr(),
        annot=True,
        cmap="coolwarm",
        linewidths=0.5
    )

    plt.title("Correlation Heatmap of Heart Disease Dataset")

    plt.tight_layout()

    plt.savefig(
        "reports/heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()