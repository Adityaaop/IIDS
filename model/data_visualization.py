import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def main() -> None:
	# Use a clean plotting style for a professional look.
	sns.set_theme(style="whitegrid", context="talk")

	# Dataset path relative to this script: model/ -> ../data/KDDTrain+_20Percent.txt
	dataset_path = Path(__file__).resolve().parent / ".." / "data" / "KDDTrain+_20Percent.txt"

	if not dataset_path.exists():
		raise FileNotFoundError(
			f"Dataset not found at: {dataset_path}. Please verify the filename in the data folder."
		)

	# The file has no header, so load with header=None.
	df = pd.read_csv(dataset_path, header=None)

	# Column index 41 (42nd column) contains the traffic label.
	df = df.rename(columns={41: "label"})

	# Create a binary traffic category for analysis.
	df["Traffic Category"] = df["label"].apply(
		lambda value: "Normal" if str(value).strip().lower() == "normal" else "Attack"
	)

	# Compute percentages for pie chart.
	category_counts = df["Traffic Category"].value_counts()
	ordered_labels = ["Normal", "Attack"]
	category_counts = category_counts.reindex(ordered_labels, fill_value=0)

	colors = ["#2E8B57", "#C0392B"]  # Green for Normal, red for Attack.

	fig, ax = plt.subplots(figsize=(8, 8))
	wedges, texts, autotexts = ax.pie(
		category_counts,
		labels=ordered_labels,
		colors=colors,
		autopct="%1.1f%%",
		startangle=90,
		counterclock=False,
		wedgeprops={"edgecolor": "white", "linewidth": 1.5},
		textprops={"fontsize": 12},
		pctdistance=0.72,
	)

	for autotext in autotexts:
		autotext.set_color("white")
		autotext.set_fontweight("bold")

	ax.set_title("Normal vs Attack Traffic Distribution", pad=20, fontweight="bold")
	ax.axis("equal")
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()
