"""Vẽ sơ đồ chatbot Action-Decision: LLM chọn action, backend tính số liệu."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "docs" / "slides" / "img" / "chatbot_flow.png"

NAVY = "#10243B"
TEAL = "#0E7C86"
GREY = "#5B6672"
LIGHT = "#ECF1F4"
plt.rcParams["font.family"] = "Arial"

fig, ax = plt.subplots(figsize=(11.2, 5.0), dpi=200)
ax.set_xlim(0, 112)
ax.set_ylim(0, 50)
ax.axis("off")


def box(x, y, w, h, text, face=LIGHT, edge=TEAL, size=9.2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.6,rounding_size=1.6",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.4,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=NAVY,
        fontweight="bold",
        linespacing=1.4,
    )


def arrow(x1, y1, x2, y2, label=None, lx=0, ly=0):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=13,
            color=NAVY,
            linewidth=1.5,
            shrinkA=2,
            shrinkB=2,
        )
    )
    if label:
        ax.text(
            (x1 + x2) / 2 + lx,
            (y1 + y2) / 2 + ly,
            label,
            ha="center",
            va="center",
            fontsize=8.2,
            color=GREY,
        )


box(1.5, 32, 17, 10, "Dock + /chat\nmessage + history", "#FFFFFF", NAVY)
box(23, 32, 18, 10, "Flask /api/chat\nvalidate input")
box(45.5, 32, 19.5, 10, "LLM decision\n1 trong 5 action", "#FFFFFF", TEAL)
box(70, 32, 22, 10, "Validator\nschema + symbol scope")

arrow(18.5, 37, 23, 37)
arrow(41, 37, 45.5, 37, "message + history", 0, 6.6)
arrow(65, 37, 70, 37, "action + args", 0, 6.6)

box(70, 13, 22, 10, "Fixed dispatcher\nhandler + prediction")
box(45.5, 13, 19.5, 10, "LLM compose\nfallback formatter", "#FFFFFF", TEAL)
box(23, 13, 18, 10, "API response\nanswer + disclaimer")
box(1.5, 13, 17, 10, "sessionStorage\ndùng chung trong tab", "#FFFFFF", NAVY)

arrow(81, 32, 81, 23, "action hợp lệ", 8.5, 0)
arrow(70, 18, 65, 18, "JSON số liệu", 0, -6.6)
arrow(45.5, 18, 41, 18, "answer", 0, -6.6)
arrow(23, 18, 18.5, 18)

ax.text(
    56,
    47,
    "Action-Decision: LLM chọn action, backend tính số liệu, LLM diễn đạt lại",
    ha="center",
    va="center",
    fontsize=10.4,
    color=NAVY,
    fontweight="bold",
)
ax.text(
    56,
    5.2,
    "Không RAG/vector DB · không tool loop · số liệu và disclaimer do backend sở hữu · compose lỗi thì giữ formatter",
    ha="center",
    va="center",
    fontsize=9.1,
    color=GREY,
)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("saved", OUT.name)
