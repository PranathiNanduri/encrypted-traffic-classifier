import pandas as pd
from pathlib import Path

# Paths
CSV_PATH = "docs/true_vs_predicted.csv"
OUT_DIR = Path("docs/error_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load predictions
df = pd.read_csv(CSV_PATH)

# Misclassified samples
errors = df[df["true_label"] != df["predicted_label"]]

# Save
errors.to_csv(OUT_DIR / "misclassified_samples.csv", index=False)

# Summary
summary = (
    errors
    .groupby(["true_label", "predicted_label"])
    .size()
    .reset_index(name="count")
)

summary.to_csv(OUT_DIR / "confusion_summary.csv", index=False)

print("\n✅ Error analysis completed")
print(summary)
