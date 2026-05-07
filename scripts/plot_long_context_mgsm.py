import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("results/processed/metrics.csv")

my_benchmarks = [
    "infinitebench_en_mc",
    "infinitebench_en_qa",
    "nih_multi_needle",
    "mgsm",
]

df = df[
    (df["benchmark"].isin(my_benchmarks)) &
    (df["metric"] == "accuracy")
]

pivot = df.pivot_table(
    index="benchmark",
    columns="model_id",
    values="value",
    aggfunc="last"
)

pivot = pivot * 100

fig, ax = plt.subplots(figsize=(12, 6))
pivot.plot(kind="bar", ax=ax)

plt.title("Accuracy on Long Context and Multilingual Benchmarks")
plt.ylabel("Accuracy (%)")  
plt.xlabel("Benchmark")
plt.xticks(rotation=30, ha="right")
plt.ylim(0, 100)  

for container in ax.containers:
    for bar in container:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,              
                f"{height:.1f}",         
                ha='center',
                va='bottom',
                fontsize=8
            )

plt.tight_layout()
plt.savefig("results/processed/assigned_benchmarks_accuracy.png", dpi=300)
plt.show()