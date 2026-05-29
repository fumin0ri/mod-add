from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import torch


def active_indices(
    mask_logits: dict[str, torch.Tensor],
    location: str,
    dim: int,
    max_nodes: int,
) -> list[int]:
    base = location.split(".")[-1]
    logits = mask_logits.get(location, mask_logits.get(base))
    if logits is None:
        return list(range(min(dim, max_nodes)))
    logits = logits.flatten()
    active = torch.nonzero(logits > 0, as_tuple=False).flatten()
    if active.numel() == 0:
        return []
    if active.numel() > max_nodes:
        values = logits[active]
        _, order = torch.topk(values, max_nodes)
        active = active[order]
    return sorted(int(i) for i in active.tolist())


def top_edges(
    weight: torch.Tensor,
    src_indices: list[int],
    tgt_indices: list[int],
    max_edges: int,
) -> list[tuple[int, int, float]]:
    if not src_indices or not tgt_indices:
        return []
    sub = weight.detach().float().cpu()[tgt_indices][:, src_indices]
    nz = torch.nonzero(sub != 0, as_tuple=False)
    if nz.numel() == 0:
        return []
    vals = sub[nz[:, 0], nz[:, 1]]
    if vals.numel() > max_edges:
        _, keep = torch.topk(vals.abs(), max_edges)
        nz = nz[keep]
        vals = vals[keep]
    edges = []
    for (tgt_pos, src_pos), value in zip(nz.tolist(), vals.tolist()):
        edges.append((src_pos, tgt_pos, float(value)))
    return edges


def render_svg(
    title: str,
    rows: list[tuple[str, list[int]]],
    edge_groups: list[tuple[int, int, list[tuple[int, int, float]]]],
) -> str:
    row_gap = 120
    node_gap = 28
    margin_x = 140
    margin_y = 70
    node_r = 7
    max_row_len = max([len(indices) for _, indices in rows] + [1])
    width = max(900, margin_x * 2 + max_row_len * node_gap)
    height = margin_y * 2 + max(1, len(rows) - 1) * row_gap

    coords: dict[tuple[int, int], tuple[float, float]] = {}
    for row_idx, (_, indices) in enumerate(rows):
        y = margin_y + row_idx * row_gap
        row_width = max(0, len(indices) - 1) * node_gap
        start_x = margin_x + (max_row_len * node_gap - row_width) / 2
        for pos, _ in enumerate(indices):
            coords[(row_idx, pos)] = (start_x + pos * node_gap, y)

    all_abs = [abs(value) for _, _, edges in edge_groups for _, _, value in edges]
    max_abs = max(all_abs) if all_abs else 1.0

    parts = [
        "<svg xmlns='http://www.w3.org/2000/svg' "
        f"width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='20' y='28' font-family='sans-serif' font-size='18'>{html.escape(title)}</text>",
    ]

    for src_row, tgt_row, edges in edge_groups:
        for src_pos, tgt_pos, value in edges:
            if (src_row, src_pos) not in coords or (tgt_row, tgt_pos) not in coords:
                continue
            x1, y1 = coords[(src_row, src_pos)]
            x2, y2 = coords[(tgt_row, tgt_pos)]
            opacity = min(0.9, max(0.08, abs(value) / max_abs))
            color = "#1f5eff" if value > 0 else "#d62728"
            parts.append(
                f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
                f"stroke='{color}' stroke-opacity='{opacity:.3f}' stroke-width='1.2'/>"
            )

    for row_idx, (row_name, indices) in enumerate(rows):
        y = margin_y + row_idx * row_gap
        parts.append(
            f"<text x='20' y='{y + 5:.1f}' font-family='monospace' font-size='13'>"
            f"{html.escape(row_name)} ({len(indices)})</text>"
        )
        for pos, index in enumerate(indices):
            x, y = coords[(row_idx, pos)]
            parts.append(
                f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{node_r}' fill='#0b57ff' stroke='black' stroke-width='0.7'>"
                f"<title>{html.escape(row_name)}[{index}]</title></circle>"
            )
            if pos % 10 == 0:
                parts.append(
                    f"<text x='{x:.1f}' y='{y + 22:.1f}' text-anchor='middle' "
                    f"font-family='monospace' font-size='9'>{index}</text>"
                )

    parts.append("</svg>")
    return "\n".join(parts)


def write_html(path: Path, title: str, svg: str, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{html.escape(title)}</title>"
            "<style>body{font-family:sans-serif;margin:24px;background:#fafafa;}"
            ".panel{background:white;border:1px solid #ddd;padding:16px;margin-bottom:24px;overflow:auto;}"
            "pre{font-size:12px;background:#f5f5f5;padding:12px;}</style></head><body>"
            f"<h1>{html.escape(title)}</h1>"
            "<div class='panel'>"
            f"{svg}"
            "</div><h2>Metadata</h2><pre>"
            f"{html.escape(json.dumps(metadata, indent=2))}</pre></body></html>"
        )


def load_prune_data(prune_masks_path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    data = torch.load(prune_masks_path, map_location="cpu")
    mask_logits = data["masker"]["logits"]
    return mask_logits, data


def mlp_circuit(
    state: dict[str, torch.Tensor],
    mask_logits: dict[str, torch.Tensor],
    layer: int,
    cfg: dict,
    max_nodes: int,
    max_edges: int,
) -> tuple[str, str, dict]:
    prefix = f"blocks.{layer}"
    rows = [
        (f"{prefix}.mlp_in", active_indices(mask_logits, f"{prefix}.mlp_in", cfg["d_model"], max_nodes)),
        (f"{prefix}.mlp_neuron", active_indices(mask_logits, f"{prefix}.mlp_neuron", cfg["d_mlp"], max_nodes)),
        (f"{prefix}.mlp_out", active_indices(mask_logits, f"{prefix}.mlp_out", cfg["d_model"], max_nodes)),
    ]
    fc_in = state[f"blocks.{layer}.mlp.fc_in.weight"]
    fc_out = state[f"blocks.{layer}.mlp.fc_out.weight"]
    edge_groups = [
        (0, 1, top_edges(fc_in, rows[0][1], rows[1][1], max_edges)),
        (1, 2, top_edges(fc_out, rows[1][1], rows[2][1], max_edges)),
    ]
    title = f"Layer {layer} MLP circuit"
    metadata = {
        "layer": layer,
        "kind": "mlp",
        "row_node_counts": {name: len(indices) for name, indices in rows},
        "edge_counts": [len(edges) for _, _, edges in edge_groups],
    }
    return title, render_svg(title, rows, edge_groups), metadata


def attention_circuit(
    state: dict[str, torch.Tensor],
    mask_logits: dict[str, torch.Tensor],
    layer: int,
    cfg: dict,
    max_nodes: int,
    max_edges: int,
) -> tuple[str, str, dict]:
    prefix = f"blocks.{layer}"
    attn_dim = cfg["n_heads"] * cfg.get("d_head", cfg["d_model"] // cfg["n_heads"])
    rows = [
        (f"{prefix}.attn_in", active_indices(mask_logits, f"{prefix}.attn_in", cfg["d_model"], max_nodes)),
        (f"{prefix}.attn_q", active_indices(mask_logits, f"{prefix}.attn_q", attn_dim, max_nodes)),
        (f"{prefix}.attn_k", active_indices(mask_logits, f"{prefix}.attn_k", attn_dim, max_nodes)),
        (f"{prefix}.attn_v", active_indices(mask_logits, f"{prefix}.attn_v", attn_dim, max_nodes)),
        (f"{prefix}.attn_out", active_indices(mask_logits, f"{prefix}.attn_out", cfg["d_model"], max_nodes)),
    ]
    edge_groups = [
        (0, 1, top_edges(state[f"blocks.{layer}.attn.W_Q.weight"], rows[0][1], rows[1][1], max_edges)),
        (0, 2, top_edges(state[f"blocks.{layer}.attn.W_K.weight"], rows[0][1], rows[2][1], max_edges)),
        (0, 3, top_edges(state[f"blocks.{layer}.attn.W_V.weight"], rows[0][1], rows[3][1], max_edges)),
        (3, 4, top_edges(state[f"blocks.{layer}.attn.W_O.weight"], rows[3][1], rows[4][1], max_edges)),
    ]
    title = f"Layer {layer} Attention circuit"
    metadata = {
        "layer": layer,
        "kind": "attention",
        "row_node_counts": {name: len(indices) for name, indices in rows},
        "edge_counts": [len(edges) for _, _, edges in edge_groups],
    }
    return title, render_svg(title, rows, edge_groups), metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prune_masks", type=str)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--max-nodes-per-location", type=int, default=80)
    parser.add_argument("--max-edges-per-pair", type=int, default=1000)
    args = parser.parse_args()

    prune_path = Path(args.prune_masks)
    mask_logits, prune_data = load_prune_data(prune_path)
    checkpoint_path = Path(args.checkpoint or prune_data["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint["config"]
    state = checkpoint["model"]
    out_dir = Path(args.out_dir) if args.out_dir else prune_path.parent / "circuit_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for layer in range(cfg["n_layers"]):
        for kind, builder in [("attn", attention_circuit), ("mlp", mlp_circuit)]:
            title, svg, metadata = builder(
                state,
                mask_logits,
                layer,
                cfg,
                args.max_nodes_per_location,
                args.max_edges_per_pair,
            )
            filename = f"layer_{layer:02d}_{kind}.html"
            write_html(out_dir / filename, title, svg, metadata)
            index.append({"file": filename, **metadata})

    with open(out_dir / "index.html", "w", encoding="utf-8") as f:
        links = "\n".join(
            f"<li><a href='{item['file']}'>{html.escape(item['file'])}</a> "
            f"{html.escape(json.dumps(item['row_node_counts']))}</li>"
            for item in index
        )
        f.write(
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Circuit visualization index</title></head><body>"
            "<h1>Circuit visualization index</h1><ul>"
            f"{links}</ul></body></html>"
        )
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"saved {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()

