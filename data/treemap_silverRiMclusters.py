import plotly.express as px
import pandas as pd

# --- Pastel palettes ---
blues = ["#BCA0FF", "#9FB9FF", "#A3E2FF"]
oranges = ["#FFC977", "#FFA7A7"]

cluster_colors = {
    "Specialization / Restriction": blues[0],
    "Generalization / Modification": blues[1],
    "Results": oranges[0],
    "Prove / Cases": oranges[1],
}

# --- Data ---
clusters = {
    "Specialization / Restriction": {
        "size": 37.83,
        "labels": [
            ("revised", 14), ("described", 13), ("investigated", 11),
            ("considered", 10), ("applied", 9), ("introduced", 8),
            ("methods", 7), ("methodology", 7), ("approach", 6),
            ("question", 5), ("problem", 4), ("related", 3),
            ("preliminary", 2), ("reviewed", 1),
        ],
    },
    "Generalization / Modification": {
        "size": 29.30,
        "labels": [
            ("generalization", 22), ("generalized", 18), ("extension", 14),
            ("modification", 12), ("variant", 10), ("analogue", 8),
            ("characterization", 6), ("extends concept", 4),
            ("generalized version", 3), ("proved generalizations", 3),
        ],
    },
    "Results": {
        "size": 21.04,
        "labels": [
            ("result", 24), ("achieved", 18), ("obtained", 15),
            ("determined", 12), ("observed", 10), ("conclusion given", 8),
            ("main result", 6), ("precise result", 4), ("result implies", 3),
        ],
    },
    "Prove / Cases": {
        "size": 11.83,
        "labels": [
            ("proof", 22), ("proved", 18), ("theorem proven", 15),
            ("provided proof", 12), ("proved cases", 10),
            ("complete proof", 8), ("claims proved", 6),
            ("proof involves", 4), ("re-proved result", 3),
            ("verify cases", 2),
        ],
    },
}

# --- Build dataframe ---
rows = []
for cluster, content in clusters.items():
    for label, val in content["labels"]:
        rows.append({
            "Cluster": cluster,
            "Label": label,
            "Value": content["size"] * (val / 100),
        })

df = pd.DataFrame(rows)

# --- Treemap ---
fig = px.treemap(
    df,
    path=["Cluster", "Label"],
    values="Value",
    color="Cluster",
    color_discrete_map=cluster_colors,
)

# --- Leaf-only text rendering ---
fig.update_traces(
    texttemplate="%{label}",   # default
    textinfo="text",
    hoverinfo="label",
    marker_line_width=1,
)

# Critical: remove text from non-leaf nodes
for trace in fig.data:
    trace.text = [
        lbl if "/" in lbl else ""   # Plotly encodes leaf paths with '/'
        for lbl in trace.labels
    ]

fig.update_layout(
    uniformtext=dict(minsize=10, mode="hide"),
    margin=dict(t=20, l=20, r=20, b=20),
    showlegend=False,
)

fig.show()
