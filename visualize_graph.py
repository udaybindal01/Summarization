"""
visualize_graph.py  —  Hypergraph Visualization for DualTowerHypergraphSummariser
==================================================================================
4-panel, publication-quality visualization of the movie hypergraph structure
built by MovieHypergraphDataset.

Panels:
    1. Role-Weighted Bipartite Layout  — entities (circles) ↔ scenes (squares)
    2. Entity Narrative Arcs           — role weight trajectory over scenes
    3. Scene-Scene Jaccard Heatmap     — coherence-loss positive pairs highlighted
    4. Entity Co-occurrence Network    — social graph (Stream 3 substrate)

Main entry points:
    plot_movie_hypergraph(...)    — save a 4-panel PNG, returns figure
    log_hypergraph_to_wandb(...)  — same, also logs to W&B
"""

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# ── Color palette ──────────────────────────────────────────────────────────────

ENTITY_TYPE_COLORS = {
    "PERSON":   "#4C72B0",
    "ORG":      "#DD8452",
    "GPE":      "#55A868",
    "FACILITY": "#C44E52",
    "OTHER":    "#8172B2",
}
_ETYPE_ID_TO_STR = {0: "PERSON", 1: "ORG", 2: "GPE", 3: "FACILITY", 4: "OTHER"}

ROLE_WEIGHT_LEVELS = {1.0: "Speaker", 0.7: "SVO Subject", 0.5: "SVO Object", 0.3: "Background"}

_BG   = "#12121F"
_GRID = "#2A2A40"
_FG   = "#E0E0F0"


def _np(x):
    if _HAS_TORCH and isinstance(x, __import__("torch").Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _etype_color(etype_ids, idx):
    if etype_ids is None:
        return ENTITY_TYPE_COLORS["OTHER"]
    eid = int(_np(etype_ids).flat[idx]) if idx < len(_np(etype_ids)) else 4
    return ENTITY_TYPE_COLORS[_ETYPE_ID_TO_STR.get(eid, "OTHER")]


def _top_entities(inc, mask, k=30):
    """Return indices of top-k entities by total role weight."""
    valid = np.where(_np(mask).astype(bool))[0]
    weights = _np(inc)[valid, :].sum(axis=1)
    order = np.argsort(weights)[::-1][:k]
    return valid[order], weights[order]


# ── Panel 1: Role-Weighted Bipartite Layout ─────────────────────────────────

def _draw_bipartite(ax, inc, entity_names, entity_type_ids, entity_mask):
    inc = _np(inc)
    N, S = inc.shape
    mask = _np(entity_mask).astype(bool) if entity_mask is not None else np.ones(N, bool)

    display_idx, total_w = _top_entities(inc, mask, k=30)
    n = len(display_idx)
    if n == 0:
        ax.text(0.5, 0.5, "No entities", transform=ax.transAxes, ha="center",
                va="center", color=_FG)
        ax.axis("off")
        return

    ent_ys   = np.linspace(0.96, 0.04, n)
    scene_ys = np.linspace(0.96, 0.04, S)

    # ── Edges ──────────────────────────────────────────────────────────────
    for ei, n_idx in enumerate(display_idx):
        for s in range(S):
            w = inc[n_idx, s]
            if w < 0.15:
                continue
            ax.plot(
                [0.12, 0.82], [ent_ys[ei], scene_ys[s]],
                color=plt.cm.YlOrRd(0.2 + w * 0.8),
                linewidth=w * 2.2,
                alpha=max(0.12, w * 0.65),
                solid_capstyle="round",
                zorder=1,
            )

    # ── Scene nodes (colored squares, temporal gradient) ───────────────────
    scene_cmap = plt.cm.cool
    for s in range(S):
        c = scene_cmap(s / max(S - 1, 1))
        rect = mpatches.FancyBboxPatch(
            (0.822, scene_ys[s] - 0.013), 0.038, 0.026,
            boxstyle="round,pad=0.003",
            facecolor=c, edgecolor="#FFFFFF44", linewidth=0.6, zorder=3,
        )
        ax.add_patch(rect)
        if S <= 24 or s % max(1, S // 12) == 0:
            ax.text(0.875, scene_ys[s], f"S{s+1}", fontsize=5,
                    va="center", color=_FG, alpha=0.75)

    # ── Entity nodes (circles, type-colored) ──────────────────────────────
    for ei, n_idx in enumerate(display_idx):
        color = _etype_color(entity_type_ids, n_idx)
        size  = 55 + total_w[ei] * 30
        ax.scatter(0.12, ent_ys[ei], s=size, c=color,
                   edgecolors="#FFFFFF66", linewidths=0.7, zorder=4)
        name = (entity_names[n_idx] if entity_names and n_idx < len(entity_names) else f"E{n_idx}")
        ax.text(0.10, ent_ys[ei], name[:16], fontsize=5.2, ha="right",
                va="center", color=color, fontweight="bold")

    # ── Legend ─────────────────────────────────────────────────────────────
    patches = [mpatches.Patch(color=c, label=t) for t, c in ENTITY_TYPE_COLORS.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=4.8,
              framealpha=0.25, ncol=2, labelcolor=_FG,
              facecolor=_BG, edgecolor=_GRID)

    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")
    ax.set_title("Role-Weighted Bipartite Hypergraph", fontsize=9,
                 fontweight="bold", color=_FG, pad=5)


# ── Panel 2: Entity Narrative Arcs ──────────────────────────────────────────

def _draw_entity_arcs(ax, inc, entity_names, entity_type_ids, entity_mask):
    inc = _np(inc)
    N, S = inc.shape
    mask = _np(entity_mask).astype(bool) if entity_mask is not None else np.ones(N, bool)

    display_idx, _ = _top_entities(inc, mask, k=20)
    if len(display_idx) == 0:
        ax.text(0.5, 0.5, "No entities", transform=ax.transAxes,
                ha="center", va="center", color=_FG)
        return

    scenes = np.arange(1, S + 1)
    used_names = set()

    for n_idx in display_idx:
        weights = inc[n_idx, :]
        if weights.max() < 0.1:
            continue
        color = _etype_color(entity_type_ids, n_idx)
        name = (entity_names[n_idx] if entity_names and n_idx < len(entity_names) else f"E{n_idx}")

        ax.plot(scenes, weights, color=color, linewidth=1.3, alpha=0.80,
                solid_capstyle="round")

        # Fill area under each arc subtly
        ax.fill_between(scenes, 0, weights, color=color, alpha=0.06)

        # Label at peak, avoid collisions
        peak_s = int(np.argmax(weights))
        label_key = (peak_s, round(weights[peak_s], 1))
        if label_key not in used_names:
            ax.text(scenes[peak_s], weights[peak_s] + 0.04, name[:12],
                    fontsize=5, color=color, ha="center", va="bottom",
                    fontweight="bold")
            used_names.add(label_key)

    # Role weight reference lines
    for w, label in ROLE_WEIGHT_LEVELS.items():
        ax.axhline(y=w, color="#888899", linewidth=0.5, linestyle="--", alpha=0.45)
        ax.text(S + 0.5, w, label, fontsize=4.5, va="center", color="#999AAB")

    ax.set_facecolor(_BG)
    ax.set_xlim(1, S + 4)
    ax.set_ylim(-0.05, 1.2)
    ax.set_xlabel("Scene Index", fontsize=7, color=_FG)
    ax.set_ylabel("Role Weight", fontsize=7, color=_FG)
    ax.tick_params(colors=_FG, labelsize=6)
    ax.grid(axis="both", color=_GRID, linewidth=0.4, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.set_title("Entity Narrative Arcs", fontsize=9, fontweight="bold",
                 color=_FG, pad=5)


# ── Panel 3: Scene-Scene Jaccard Heatmap ─────────────────────────────────────

def _draw_scene_jaccard(ax, inc, entity_mask):
    inc = _np(inc)
    N, S = inc.shape
    mask = _np(entity_mask).astype(bool) if entity_mask is not None else np.ones(N, bool)
    inc_valid = inc[mask]

    # Vectorised Jaccard
    inter = inc_valid.T @ inc_valid                                        # [S, S]
    sums  = inc_valid.sum(axis=0, keepdims=True)                          # [1, S]
    union = sums + sums.T - inter
    jac   = inter / np.maximum(union, 1e-8)
    np.fill_diagonal(jac, np.nan)

    cmap = sns.color_palette("mako", as_cmap=True)
    sns.heatmap(
        jac, ax=ax, cmap=cmap, vmin=0, vmax=0.8,
        xticklabels=False, yticklabels=False,
        cbar_kws={"shrink": 0.65, "label": "Jaccard"},
        linewidths=0,
    )
    ax.collections[0].colorbar.ax.tick_params(colors=_FG, labelsize=6)
    ax.collections[0].colorbar.ax.yaxis.label.set_color(_FG)

    # Overlay contour for positive coherence pairs (threshold = 0.25)
    pos = (np.nan_to_num(jac) > 0.25).astype(float)
    np.fill_diagonal(pos, 0)
    if pos.sum() > 0:
        ax.contour(np.arange(S) + 0.5, np.arange(S) + 0.5, pos,
                   levels=[0.5], colors=["#FF6030"], linewidths=0.7, alpha=0.9)

    ax.set_title("Scene-Scene Jaccard Similarity\n"
                 "(red contour = coherence-loss positive pairs)",
                 fontsize=8, fontweight="bold", color=_FG, pad=5)
    ax.set_xlabel("Scene", fontsize=7, color=_FG)
    ax.set_ylabel("Scene", fontsize=7, color=_FG)
    ax.tick_params(colors=_FG)
    ax.set_facecolor(_BG)


# ── Panel 4: Entity Co-occurrence Network (Stream 3 substrate) ───────────────

def _draw_cooccurrence_network(ax, inc, entity_names, entity_type_ids, entity_mask):
    inc = _np(inc)
    N, S = inc.shape
    mask = _np(entity_mask).astype(bool) if entity_mask is not None else np.ones(N, bool)

    display_idx, total_w = _top_entities(inc, mask, k=25)
    n = len(display_idx)
    if n < 2:
        ax.text(0.5, 0.5, "Not enough entities", transform=ax.transAxes,
                ha="center", va="center", color=_FG, fontsize=7)
        ax.axis("off")
        ax.set_title("Entity Co-occurrence Network (Stream 3)",
                     fontsize=9, fontweight="bold", color=_FG, pad=5)
        return

    # Co-occurrence = sum_s (w_i,s * w_j,s)
    sub_inc = inc[display_idx, :]            # [n, S]
    cooccur = sub_inc @ sub_inc.T            # [n, n]
    np.fill_diagonal(cooccur, 0)

    G = nx.Graph()
    for i, n_idx in enumerate(display_idx):
        name = (entity_names[n_idx] if entity_names and n_idx < len(entity_names) else f"E{n_idx}")
        G.add_node(i, label=name[:14],
                   etype=int(_np(entity_type_ids).flat[n_idx]) if entity_type_ids is not None else 4,
                   tw=float(total_w[i]))

    threshold = cooccur.max() * 0.06
    for i in range(n):
        for j in range(i + 1, n):
            if cooccur[i, j] > threshold:
                G.add_edge(i, j, weight=float(cooccur[i, j]))

    if len(G.edges) == 0:
        ax.text(0.5, 0.5, "No co-occurrence edges above threshold",
                transform=ax.transAxes, ha="center", va="center",
                color=_FG, fontsize=7)
        ax.axis("off")
        ax.set_title("Entity Co-occurrence Network (Stream 3)",
                     fontsize=9, fontweight="bold", color=_FG, pad=5)
        return

    try:
        pos = nx.kamada_kawai_layout(G, weight="weight")
    except Exception:
        pos = nx.spring_layout(G, seed=42, k=2.2 / max(n ** 0.5, 1))

    node_colors = [ENTITY_TYPE_COLORS[_ETYPE_ID_TO_STR.get(G.nodes[i]["etype"], "OTHER")]
                   for i in G.nodes]
    node_sizes  = [80 + G.nodes[i]["tw"] * 70 for i in G.nodes]

    edge_weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    max_ew = edge_weights.max() if len(edge_weights) > 0 else 1.0
    edge_widths = (edge_weights / max_ew * 3.5).tolist()
    edge_colors = [plt.cm.Blues(0.3 + 0.7 * (w / max_ew)) for w in edge_weights]

    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                           edge_color=edge_colors, alpha=0.65)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, edgecolors="#FFFFFF55",
                           linewidths=0.7)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={i: G.nodes[i]["label"] for i in G.nodes},
                            font_size=5, font_color="white", font_weight="bold")

    # Entity-type legend
    shown_types = {_ETYPE_ID_TO_STR.get(G.nodes[i]["etype"], "OTHER") for i in G.nodes}
    patches = [mpatches.Patch(color=ENTITY_TYPE_COLORS[t], label=t) for t in shown_types]
    ax.legend(handles=patches, loc="lower right", fontsize=4.8,
              framealpha=0.25, labelcolor=_FG, facecolor=_BG, edgecolor=_GRID)

    ax.axis("off")
    ax.set_facecolor(_BG)
    ax.set_title("Entity Co-occurrence Network\n(Stream 3 substrate)",
                 fontsize=9, fontweight="bold", color=_FG, pad=5)


# ── Public API ──────────────────────────────────────────────────────────────────

def plot_movie_hypergraph(
    incidence_matrix,
    entity_names,
    entity_type_ids=None,
    entity_mask=None,
    movie_name="Movie",
    save_path="hypergraph.png",
):
    """
    Build and save a 4-panel hypergraph visualization.

    Args:
        incidence_matrix : [N, S] tensor or ndarray  (float role weights)
        entity_names     : List[str] of length N  (empty string for padding)
        entity_type_ids  : [N] int tensor/array (0=PERSON…4=OTHER), or None
        entity_mask      : [N] bool tensor/array, or None (treat all as valid)
        movie_name       : title string
        save_path        : output PNG path  (None = no save)

    Returns:
        matplotlib Figure
    """
    inc = _np(incidence_matrix)
    N, S = inc.shape

    if entity_mask is None:
        entity_mask = np.ones(N, dtype=bool)

    fig = plt.figure(figsize=(20, 16), facecolor=_BG)
    n_valid = int(_np(entity_mask).astype(bool).sum())
    fig.suptitle(
        f"Hypergraph Structure  —  {movie_name}\n"
        f"{n_valid} entities  ×  {S} scenes",
        fontsize=13, fontweight="bold", color=_FG, y=0.985,
    )

    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.22,
                          left=0.04, right=0.98, top=0.93, bottom=0.04)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    for ax in axes:
        ax.set_facecolor(_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID)
        ax.tick_params(colors=_FG)

    _draw_bipartite(axes[0], inc, entity_names, entity_type_ids, entity_mask)
    _draw_entity_arcs(axes[1], inc, entity_names, entity_type_ids, entity_mask)
    _draw_scene_jaccard(axes[2], inc, entity_mask)
    _draw_cooccurrence_network(axes[3], inc, entity_names, entity_type_ids, entity_mask)

    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved hypergraph visualization → {save_path}")

    return fig


def log_hypergraph_to_wandb(
    incidence_matrix,
    entity_names,
    entity_type_ids=None,
    entity_mask=None,
    movie_name="",
    step=None,
    save_path=None,
):
    """
    Build the 4-panel visualization and log it to W&B.
    No-ops gracefully if wandb is not initialized.
    """
    try:
        import wandb as _wandb
        if not _wandb.run:
            return
    except ImportError:
        return

    fig = plot_movie_hypergraph(
        incidence_matrix, entity_names,
        entity_type_ids=entity_type_ids,
        entity_mask=entity_mask,
        movie_name=movie_name,
        save_path=save_path,
    )
    log_dict = {"hypergraph/structure": _wandb.Image(fig)}
    if step is not None:
        log_dict["epoch"] = step
    _wandb.log(log_dict)
    plt.close(fig)
