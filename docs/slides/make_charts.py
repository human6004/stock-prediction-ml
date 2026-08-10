"""Ve chart cho slide tu du lieu thuc trong reports/. Chi doc, khong sua repo."""
import json, shutil
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[2]
REP = BASE / "reports"
IMG = Path(__file__).resolve().parent / "img"
IMG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
    "axes.edgecolor": "#4A5568",
    "axes.labelcolor": "#1A202C",
    "text.color": "#1A202C",
    "figure.facecolor": "white",
})
NAVY = "#1B3A5C"
TEAL = "#2C7A7B"
AMBER = "#B7791F"
GRAY = "#A0AEC0"
RED = "#9B2C2C"

summary = json.loads((REP / "pipeline_summary.json").read_text(encoding="utf-8"))

def bar_labels(ax, bars, fmt="{:,.0f}", dy=0.01):
    top = max(b.get_height() for b in bars)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * dy,
                fmt.format(b.get_height()), ha="center", va="bottom", fontsize=11)

# 1. Data funnel
ds, cl, fe, lb = summary["dataset_report"], summary["clean_report"], summary["feature_report"], summary["label_report"]
stages = ["Raw OHLCV", "Sau lam sach", "Sau loc ma", "Sau 20 feature", "Sau gan nhan"]
vals = [ds["rows"], cl["rows_after_cleaning"], cl["rows_after_symbol_filter"],
        fe["rows_after_features"], lb["rows_after_labeling"]]
fig, ax = plt.subplots(figsize=(10, 4.6))
bars = ax.bar(stages, vals, color=[NAVY, NAVY, TEAL, TEAL, AMBER], width=0.62)
bar_labels(ax, bars)
ax.set_ylabel("So dong")
ax.set_ylim(0, max(vals) * 1.15)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#E2E8F0")
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(IMG / "data_funnel.png", dpi=200)
plt.close(fig)

# 2. Label balance
fig, ax = plt.subplots(figsize=(5.4, 4.2))
bars = ax.bar(["NOT_UP", "UP"], [lb["not_up_count"], lb["up_count"]], color=[GRAY, TEAL], width=0.5)
bar_labels(ax, bars)
ax.set_ylabel("So dong co nhan")
ax.set_ylim(0, lb["not_up_count"] * 1.18)
ax.set_title(f"UP = {lb['up_ratio']*100:.1f}% tong the", fontsize=12)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#E2E8F0"); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(IMG / "label_balance.png", dpi=200); plt.close(fig)

# 3. Split rows + dates
sp = summary["split_report"]
labels = [f"TRAIN\n{sp['train_date_min']} - {sp['train_date_max']}",
          f"VALIDATION\n{sp['validation_date_min']} - {sp['validation_date_max']}",
          f"TEST (khoa)\n{sp['test_date_min']} - {sp['test_date_max']}"]
vals = [sp["train_rows"], sp["validation_rows"], sp["test_rows"]]
fig, ax = plt.subplots(figsize=(10, 4.4))
bars = ax.bar(labels, vals, color=[NAVY, TEAL, AMBER], width=0.55)
bar_labels(ax, bars)
ax.set_ylabel("So dong")
ax.set_ylim(0, max(vals) * 1.18)
ax.set_title(f"Purge {sp['purged_train_validation_rows']:,} dong TRAIN/VAL va "
             f"{sp['purged_validation_test_rows']:,} dong VAL/TEST", fontsize=12)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#E2E8F0"); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(IMG / "split_rows.png", dpi=200); plt.close(fig)

# 4. CV F1_UP on TRAIN with std
tun = pd.read_csv(REP / "tuning_results.csv")
fig, ax = plt.subplots(figsize=(8.6, 4.4))
bars = ax.bar(tun["model_name"], tun["cv_f1_up"], yerr=tun["cv_f1_up_std"], capsize=6,
              color=[GRAY, NAVY, GRAY], width=0.5, ecolor="#4A5568")
for b, v, s in zip(bars, tun["cv_f1_up"], tun["cv_f1_up_std"]):
    ax.text(b.get_x() + b.get_width() / 2, v + s + 0.012, f"{v:.4f}", ha="center", fontsize=11)
ax.set_ylabel("CV F1_UP (4 fold, purge 5 phien)")
ax.set_ylim(0, 0.62)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#E2E8F0"); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(IMG / "cv_f1_up.png", dpi=200); plt.close(fig)

# 5. VALIDATION selection vs Always UP
cmp = pd.read_csv(REP / "model_comparison.csv")
rows = cmp[cmp["model_name"].isin(["Logistic Regression", "Random Forest", "Gradient Boosting", "Always UP"])]
colors = [NAVY if n == "Random Forest" else (RED if n == "Always UP" else GRAY) for n in rows["model_name"]]
fig, ax = plt.subplots(figsize=(9.4, 4.4))
bars = ax.bar(rows["model_name"], rows["f1_up"], color=colors, width=0.52)
bar_labels(ax, bars, fmt="{:.4f}")
ax.set_ylabel("F1_UP tren VALIDATION")
ax.set_ylim(0, 0.63)
ax.axhline(float(rows.loc[rows["model_name"] == "Always UP", "f1_up"].iloc[0]),
           color=RED, ls="--", lw=1.4)
ax.set_title("Random Forest cao nhat trong 3 model, van duoi baseline Always UP", fontsize=12)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#E2E8F0"); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(IMG / "validation_f1_up.png", dpi=200); plt.close(fig)

# 6. TEST vs baselines: F1_UP + accuracy
fin = pd.read_csv(REP / "final_model_evaluation.csv")
names = list(fin["model_name"])
x = range(len(names))
fig, ax = plt.subplots(figsize=(9.4, 4.4))
w = 0.36
b1 = ax.bar([i - w/2 for i in x], fin["f1_up"], width=w, color=NAVY, label="F1_UP")
b2 = ax.bar([i + w/2 for i in x], fin["accuracy"], width=w, color=TEAL, label="Accuracy")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.012, f"{b.get_height():.4f}",
                ha="center", fontsize=10)
ax.set_xticks(list(x)); ax.set_xticklabels(names)
ax.set_ylim(0, 1.0); ax.set_ylabel("Gia tri tren TEST (20.350 dong)")
ax.legend(frameon=False, ncol=2, loc="upper center")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#E2E8F0"); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(IMG / "test_vs_baseline.png", dpi=200); plt.close(fig)

# 7. Confusion matrix TEST
cm = pd.read_csv(REP / "confusion_matrix.csv", index_col=0)
fig, ax = plt.subplots(figsize=(5.8, 4.6))
im = ax.imshow(cm.values, cmap="Blues")
ax.set_xticks([0, 1], ["pred NOT_UP", "pred UP"])
ax.set_yticks([0, 1], ["thuc NOT_UP", "thuc UP"])
vmax = cm.values.max()
for i in range(2):
    for j in range(2):
        v = cm.values[i, j]
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=15,
                color="white" if v > vmax * 0.55 else "#1A202C")
ax.set_title("Confusion matrix TEST", fontsize=12)
fig.tight_layout(); fig.savefig(IMG / "confusion_matrix_test.png", dpi=200); plt.close(fig)

# 8. Feature importance top 8
fi = pd.read_csv(REP / "feature_importance.csv").head(8).iloc[::-1]
fig, ax = plt.subplots(figsize=(8.8, 4.4))
bars = ax.barh(fi["feature"], fi["importance"], color=NAVY, height=0.62)
for b, v in zip(bars, fi["importance"]):
    ax.text(v + 0.004, b.get_y() + b.get_height()/2, f"{v:.3f}", va="center", fontsize=11)
ax.set_xlabel("Gini importance (Random Forest)")
ax.set_xlim(0, fi["importance"].max() * 1.18)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color="#E2E8F0"); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(IMG / "feature_importance_top8.png", dpi=200); plt.close(fig)

# 9. CV fold spread per model
fold = pd.read_csv(REP / "cv_fold_results.csv")
fig, ax = plt.subplots(figsize=(9.4, 4.4))
for name, color, mk in [("Logistic Regression", GRAY, "o"), ("Random Forest", NAVY, "s"),
                        ("Gradient Boosting", TEAL, "^")]:
    sub = fold[fold["model_name"] == name]
    ax.plot(sub["fold"], sub["f1_up"], marker=mk, color=color, lw=1.8, label=name)
ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("Fold (theo thoi gian)")
ax.set_ylabel("F1_UP moi fold"); ax.set_ylim(0.38, 0.55)
ax.legend(frameon=False, fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color="#E2E8F0"); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(IMG / "cv_fold_spread.png", dpi=200); plt.close(fig)

# copy san co
shutil.copyfile(BASE / "docs/report_assets/architecture_overview.png", IMG / "architecture_overview.png")
shutil.copyfile(BASE / "docs/report_assets/dataflow_pipeline.png", IMG / "dataflow_pipeline.png")
shutil.copyfile(BASE / "docs/report_assets/logo_ctu.png", IMG / "logo_ctu.png")
print("charts:", sorted(p.name for p in IMG.iterdir()))
