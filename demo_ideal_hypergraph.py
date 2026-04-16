"""
demo_ideal_hypergraph.py
========================
Generates a side-by-side comparison of BAD (dense, noisy) vs GOOD (sparse,
structured) hypergraph incidence matrices so you can visually see what the
training should converge toward.

Run:
    python demo_ideal_hypergraph.py

Saves:  ideal_vs_bad_hypergraph.png  in the current directory.

What a GOOD hypergraph looks like
----------------------------------
Think of a movie like "Young Guns":
  - Billy the Kid appears in scenes 1-40 (protagonist, high weight)
  - Doc Scurlock appears in scenes 1-35 (friend, follows protagonist)
  - Murphy appears in scenes 5-20 and 38-45 (antagonist, two act blocks)
  - Sheriff appears only in scenes 35-45 (late-game authority figure)
  - Random townsfolk appear in 1-3 scenes each (bit parts)

The incidence matrix should look like a few diagonal/block-sparse bands —
NOT a uniform gray rectangle.

Panels:
  Row 1 (BAD):  dense matrix, flat narrative arcs, structureless Jaccard,
                one giant co-occurrence blob
  Row 2 (GOOD): sparse matrix with clear blocks, distinct arcs, block-diagonal
                Jaccard (acts), small tight communities in co-occurrence
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx

np.random.seed(42)

# ── Colour palette (matches visualize_graph.py) ────────────────────────────
_BG   = "#12121F"
_GRID = "#2A2A40"
_FG   = "#E0E0F0"

COLORS = {
    "protagonist":  "#4C72B0",
    "antagonist":   "#C44E52",
    "sidekick":     "#55A868",
    "love_int":     "#DD8452",
    "authority":    "#8172B2",
    "minor":        "#888899",
}

# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data factories
# ──────────────────────────────────────────────────────────────────────────────

N_ENT, N_SCN = 20, 48

def _bad_incidence():
    """Dense matrix: most entities appear in most scenes at ~0.3 weight."""
    inc = np.random.uniform(0.25, 0.45, (N_ENT, N_SCN))
    inc += np.random.normal(0, 0.08, (N_ENT, N_SCN))
    inc = np.clip(inc, 0.1, 0.5)
    return inc

def _good_incidence():
    """
    Structured sparse matrix mimicking a 3-act screenplay.

    Act 1  scenes  0-15  (intro + rising tension)
    Act 2  scenes 16-31  (confrontation)
    Act 3  scenes 32-47  (climax + resolution)
    """
    inc = np.zeros((N_ENT, N_SCN))

    def _arc(entity_idx, active_scenes, base_weight, jitter=0.08, color=None):
        for s in active_scenes:
            w = base_weight + np.random.normal(0, jitter)
            inc[entity_idx, s] = float(np.clip(w, 0.1, 1.0))

    # Protagonist — present throughout, speaker weight 1.0
    _arc(0,  range(0, 48),        0.85, 0.10)   # Billy the Kid

    # Main sidekick — acts 1 and 2, fades out
    _arc(1,  range(0, 36),        0.70, 0.10)   # Doc Scurlock
    _arc(1,  range(36, 48),       0.30, 0.08)   # occasional mention late

    # Antagonist — appears in two blocks (act 1 threat, act 3 showdown)
    _arc(2,  range(4, 18),        0.80, 0.08)   # Murphy — act 1
    _arc(2,  range(36, 48),       0.90, 0.07)   # Murphy — act 3 climax

    # Love interest — act 1 and early act 2
    _arc(3,  range(6, 25),        0.65, 0.12)

    # Authority figure — only enters in act 3
    _arc(4,  range(30, 48),       0.75, 0.10)   # Sheriff

    # Four minor characters with limited scene ranges
    _arc(5,  range(0, 12),        0.55, 0.12)   # opens movie
    _arc(6,  range(14, 26),       0.50, 0.10)   # mid-film
    _arc(7,  range(22, 38),       0.45, 0.10)   # bridge between acts
    _arc(8,  range(40, 48),       0.60, 0.10)   # final act only

    # Very minor / cameo entities (appear in 1-3 scenes only)
    for i in range(9, 20):
        n_scenes = np.random.randint(1, 5)
        start = np.random.randint(0, N_SCN - n_scenes)
        scenes = range(start, start + n_scenes)
        _arc(i, scenes, np.random.uniform(0.3, 0.6), 0.05)

    return inc


# ──────────────────────────────────────────────────────────────────────────────
# Panel drawing helpers
# ──────────────────────────────────────────────────────────────────────────────

_ENTITY_LABELS_GOOD = [
    "Billy", "Doc", "Murphy", "Kate", "Sheriff",
    "McSween", "Chavez", "Brewer", "McCloskey", "Bowdre",
    "Deputy1", "Bounty1", "Rancher", "Barkeep", "Deputy2",
    "Guard", "Cowboy", "Townsfolk1", "Townsfolk2", "Townsfolk3",
]
_ENTITY_LABELS_BAD = [
    "billy", "doc", "car", "door", "gun", "room", "house", "phone",
    "time", "thing", "place", "money", "back", "face", "man",
    "way", "hand", "life", "night", "world",
]

_ROLE_WEIGHTS = {1.0: "Speaker", 0.7: "Subject", 0.5: "Object", 0.3: "Background"}


def _draw_incidence(ax, inc, title, cmap="YlOrRd", annotate_cols=True):
    ax.set_facecolor(_BG)
    im = ax.imshow(inc, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_xlabel("Scene →", fontsize=7, color=_FG)
    ax.set_ylabel("Entity", fontsize=7, color=_FG)
    ax.tick_params(colors=_FG, labelsize=5)
    ax.set_title(title, fontsize=8, fontweight="bold", color=_FG, pad=4)
    cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.ax.tick_params(colors=_FG, labelsize=5)
    cb.set_label("Role weight", color=_FG, fontsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    return im


def _draw_arcs(ax, inc, labels, title):
    ax.set_facecolor(_BG)
    scenes = np.arange(1, inc.shape[1] + 1)
    palette = plt.cm.tab20.colors
    for i in range(inc.shape[0]):
        w = inc[i]
        if w.max() < 0.15:
            continue
        ax.plot(scenes, w, color=palette[i % 20], linewidth=1.2, alpha=0.8)
        ax.fill_between(scenes, 0, w, color=palette[i % 20], alpha=0.05)
        peak = int(np.argmax(w))
        ax.text(scenes[peak], w[peak] + 0.05, labels[i][:10],
                fontsize=4.5, color=palette[i % 20], ha="center", fontweight="bold")
    for w_ref, lbl in _ROLE_WEIGHTS.items():
        ax.axhline(y=w_ref, color="#888899", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.text(scenes[-1] + 0.5, w_ref, lbl, fontsize=4, va="center", color="#999AAB")
    ax.set_facecolor(_BG)
    ax.set_xlim(1, scenes[-1] + 5)
    ax.set_ylim(-0.05, 1.25)
    ax.set_xlabel("Scene →", fontsize=7, color=_FG)
    ax.set_ylabel("Role weight", fontsize=7, color=_FG)
    ax.tick_params(colors=_FG, labelsize=6)
    ax.grid(color=_GRID, linewidth=0.4, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.set_title(title, fontsize=8, fontweight="bold", color=_FG, pad=4)


def _draw_jaccard(ax, inc, title):
    ax.set_facecolor(_BG)
    inc_f = inc.astype(float)
    inter = inc_f.T @ inc_f
    sums  = inc_f.sum(0, keepdims=True)
    union = sums + sums.T - inter
    jac   = inter / np.maximum(union, 1e-8)
    np.fill_diagonal(jac, np.nan)
    cmap = sns.color_palette("mako", as_cmap=True)
    sns.heatmap(jac, ax=ax, cmap=cmap, vmin=0, vmax=1,
                xticklabels=False, yticklabels=False,
                cbar_kws={"shrink": 0.65, "label": "Jaccard"},
                linewidths=0)
    ax.collections[0].colorbar.ax.tick_params(colors=_FG, labelsize=6)
    ax.collections[0].colorbar.ax.yaxis.label.set_color(_FG)
    # Positive coherence pairs (threshold 0.25)
    pos = (np.nan_to_num(jac) > 0.25).astype(float)
    np.fill_diagonal(pos, 0)
    S = jac.shape[0]
    if pos.sum() > 0:
        ax.contour(np.arange(S) + 0.5, np.arange(S) + 0.5, pos,
                   levels=[0.5], colors=["#FF6030"], linewidths=0.8, alpha=0.9)
    ax.set_title(title, fontsize=8, fontweight="bold", color=_FG, pad=4)
    ax.set_xlabel("Scene", fontsize=7, color=_FG)
    ax.set_ylabel("Scene", fontsize=7, color=_FG)
    ax.tick_params(colors=_FG)


def _draw_cooccur(ax, inc, labels, title):
    ax.set_facecolor(_BG)
    # top 15 by total weight
    totals = inc.sum(1)
    top_idx = np.argsort(totals)[::-1][:15]
    sub = inc[top_idx]
    co  = sub @ sub.T
    np.fill_diagonal(co, 0)

    G = nx.Graph()
    for i, gi in enumerate(top_idx):
        G.add_node(i, label=labels[gi][:12], w=float(totals[gi]))

    thresh = co.max() * 0.08 if co.max() > 0 else 0
    for i in range(len(top_idx)):
        for j in range(i + 1, len(top_idx)):
            if co[i, j] > thresh:
                G.add_edge(i, j, weight=float(co[i, j]))

    if len(G.edges) == 0:
        ax.text(0.5, 0.5, "No edges", transform=ax.transAxes,
                ha="center", va="center", color=_FG)
        ax.axis("off")
        ax.set_title(title, fontsize=8, fontweight="bold", color=_FG, pad=4)
        return

    try:
        pos = nx.kamada_kawai_layout(G, weight="weight")
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    palette = plt.cm.tab20.colors
    node_colors = [palette[i % 20] for i in range(len(G.nodes))]
    node_sizes  = [80 + G.nodes[i]["w"] * 60 for i in G.nodes]
    ews = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    max_ew = ews.max() if len(ews) > 0 else 1.0
    nx.draw_networkx_edges(G, pos, ax=ax,
                           width=(ews / max_ew * 3.0).tolist(),
                           edge_color=[plt.cm.Blues(0.4 + 0.6 * w / max_ew) for w in ews],
                           alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, edgecolors="#FFFFFF55", linewidths=0.6)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={i: G.nodes[i]["label"] for i in G.nodes},
                            font_size=5, font_color="white", font_weight="bold")
    ax.axis("off")
    ax.set_title(title, fontsize=8, fontweight="bold", color=_FG, pad=4)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    bad  = _bad_incidence()
    good = _good_incidence()

    fig = plt.figure(figsize=(22, 14), facecolor=_BG)
    fig.suptitle(
        "Hypergraph Quality: BAD (dense/noisy)  vs  GOOD (sparse/structured)\n"
        "Your model's graph should look like the GOOD row after training.",
        fontsize=12, fontweight="bold", color=_FG, y=0.99,
    )

    gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.28,
                          left=0.04, right=0.99, top=0.92, bottom=0.04)

    # ── BAD row ───────────────────────────────────────────────────────────────
    ax_bad_inc  = fig.add_subplot(gs[0, 0])
    ax_bad_arc  = fig.add_subplot(gs[0, 1])
    ax_bad_jac  = fig.add_subplot(gs[0, 2])
    ax_bad_co   = fig.add_subplot(gs[0, 3])

    _draw_incidence(ax_bad_inc, bad,
                    "❌  BAD: Incidence Matrix\n(dense, all entities in all scenes)")
    _draw_arcs(ax_bad_arc, bad, _ENTITY_LABELS_BAD,
               "❌  BAD: Narrative Arcs\n(flat, no character trajectory)")
    _draw_jaccard(ax_bad_jac, bad,
                  "❌  BAD: Scene Similarity\n(uniformly high, no structure)")
    _draw_cooccur(ax_bad_co, bad, _ENTITY_LABELS_BAD,
                  "❌  BAD: Co-occurrence Network\n(one dense blob, no communities)")

    # ── GOOD row ──────────────────────────────────────────────────────────────
    ax_good_inc = fig.add_subplot(gs[1, 0])
    ax_good_arc = fig.add_subplot(gs[1, 1])
    ax_good_jac = fig.add_subplot(gs[1, 2])
    ax_good_co  = fig.add_subplot(gs[1, 3])

    _draw_incidence(ax_good_inc, good,
                    "✓  GOOD: Incidence Matrix\n(sparse blocks, protagonist + acts visible)")
    _draw_arcs(ax_good_arc, good, _ENTITY_LABELS_GOOD,
               "✓  GOOD: Narrative Arcs\n(distinct arcs, characters enter/exit)")
    _draw_jaccard(ax_good_jac, good,
                  "✓  GOOD: Scene Similarity\n(block-diagonal = narrative acts)")
    _draw_cooccur(ax_good_co, good, _ENTITY_LABELS_GOOD,
                  "✓  GOOD: Co-occurrence Network\n(tight communities, protagonist hub)")

    # ── Annotation callouts ───────────────────────────────────────────────────
    fig.text(0.01, 0.71, "BAD\nrun", fontsize=11, color="#C44E52",
             fontweight="bold", va="center",
             bbox=dict(facecolor="#2A0A0A", edgecolor="#C44E52",
                       boxstyle="round,pad=0.4", alpha=0.8))
    fig.text(0.01, 0.28, "GOOD\nrun", fontsize=11, color="#55A868",
             fontweight="bold", va="center",
             bbox=dict(facecolor="#0A2A0A", edgecolor="#55A868",
                       boxstyle="round,pad=0.4", alpha=0.8))

    out = "ideal_vs_bad_hypergraph.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    plt.close(fig)

    # ── Print interpretation guide ─────────────────────────────────────────────
    print("""
Interpretation guide
====================

Incidence matrix (panel 1)
  GOOD: sparse, bright horizontal bands for protagonist + major characters,
        dark gaps where characters are absent. You should see ~3-8 bright
        rows max, rest near-zero.
  BAD:  uniform gray/yellow rectangle — every entity in every scene.

Narrative arcs (panel 2)
  GOOD: each character has a unique trajectory. Protagonist stays high,
        antagonist spikes during confrontation, minor chars peak once.
  BAD:  all lines flat at ~0.3 (just background noise).

Scene-scene Jaccard (panel 3)
  GOOD: block-diagonal structure — Act 1 scenes share entities with each
        other, Act 2 with each other, etc. Red contour = coherence-loss
        positive pairs (correctly grouped nearby scenes).
  BAD:  uniform high similarity everywhere — no act structure.

Co-occurrence network (panel 4)
  GOOD: protagonist node is central hub; distinct sub-communities around
        antagonist, love interest, etc. Sparse edges with clear clusters.
  BAD:  one giant fully-connected blob — no social structure.

W&B metrics to watch (added in latest train.py)
  hg_ents_per_scene   target: 3-10      (bad: >20)
  hg_scenes_per_ent   target: 3-15      (bad: >40, means entity appears everywhere)
  hg_entity_coverage  target: >0.30     (do tracked entities appear in summaries?)
  fusion_gate_mean    target: 0.3-0.7   (bad: ~1.0 = hypergraph ignored by decoder)
""")


if __name__ == "__main__":
    main()
