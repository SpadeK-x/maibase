import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


AXIS_METRIC_GROUPS: Dict[str, List[str]] = {
    "density_load": [
        "events_per_second",
        "mean_density_500ms",
        "p90_density_500ms",
        "busy_density_mean",
        "busy_density_p90",
        "slide_density_mean",
        "slide_density_p90",
    ],
    "rhythm_complexity": [
        "interval_cv",
        "interval_entropy",
        "rhythm_switch_ratio",
        "short_long_alternation_ratio",
        "low_density_rhythm_switch_ratio",
        "nonburst_rhythm_switch_ratio",
    ],
    "movement_stretch": [
        "mean_outer_move_dist",
        "outer_move_p90",
        "outer_move_p95",
        "outer_move_ge_0_25_ratio",
        "outer_move_ge_0_375_ratio",
        "outer_move_ge_0_5_ratio",
        "busy_outer_move_mean",
        "busy_outer_move_p90",
        "dual_outer_span_mean",
        "dual_outer_span_p90",
        "dual_outer_wide_ratio",
        "span_jump_mean",
        "span_jump_p90",
        "span_jump_ge_0_5_ratio",
    ],
    "occupancy_pressure": [
        "busy_ratio",
        "hold_active_ratio",
        "slide_active_ratio",
        "hold_only_ratio",
        "slide_only_ratio",
        "active_to_tap_ratio",
        "active_to_compound_ratio",
    ],
    "slide_complexity": [
        "mean_slide_span",
        "max_slide_span",
        "slide_span_p90",
        "long_slide_ratio",
        "slide_span_jump_p90",
        "slide_compound_ratio",
        "compound_with_slide_ratio",
        "slide_conflict_ratio",
        "slide_conflict_when_busy_ratio",
        "slide_conflict_active_ratio",
        "slide_tap_interrupt_ratio",
    ],
    "touch_cross_mix": [
        "touch_ratio",
        "compound_ratio",
        "cross_zone_ratio",
        "inner_add_ge2_ratio",
        "inner_count_ge2_ratio",
        "busy_cross_zone_ratio",
        "busy_inner_add_ge2_ratio",
        "dual_outer_ratio",
    ],
}


def round_float(value: float) -> float:
    return round(float(value), 6)


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_feature_matrix(rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    if not rows:
        raise ValueError("No rows found in style feature CSV.")

    reserved = {"chart", "bucket_label", "raw_level"}
    metric_names = [name for name in rows[0].keys() if name not in reserved]
    charts = [row["chart"] for row in rows]
    bucket_labels = [row["bucket_label"] for row in rows]
    raw_levels = [float(row["raw_level"]) for row in rows]
    matrix = torch.tensor(
        [[float(row[name]) for name in metric_names] for row in rows],
        dtype=torch.float32,
    )
    return {
        "metric_names": metric_names,
        "charts": charts,
        "bucket_labels": bucket_labels,
        "raw_levels": raw_levels,
        "matrix": matrix,
    }


def compute_axis_scores(metric_names: Sequence[str], scaled_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
    name_to_index = {name: idx for idx, name in enumerate(metric_names)}
    axis_scores: Dict[str, torch.Tensor] = {}
    for axis_name, metrics in AXIS_METRIC_GROUPS.items():
        indices = [name_to_index[name] for name in metrics if name in name_to_index]
        if not indices:
            raise KeyError(f"No metrics found for axis {axis_name}")
        axis_scores[axis_name] = scaled_matrix[:, indices].mean(dim=1)
    return axis_scores


def build_cluster_input(
    metric_names: Sequence[str],
    scaled_matrix: torch.Tensor,
    cluster_space: str,
) -> Dict[str, object]:
    if cluster_space == "full":
        return {
            "space_names": list(metric_names),
            "space_matrix": scaled_matrix,
            "axis_scores": compute_axis_scores(metric_names, scaled_matrix),
        }

    axis_scores = compute_axis_scores(metric_names, scaled_matrix)
    axis_names = list(AXIS_METRIC_GROUPS.keys())
    axis_matrix = torch.stack([axis_scores[name] for name in axis_names], dim=1)
    return {
        "space_names": axis_names,
        "space_matrix": axis_matrix,
        "axis_scores": axis_scores,
    }


def representative_indices(
    space_matrix: torch.Tensor,
    cluster_ids: Sequence[int],
    centroids: torch.Tensor,
) -> Dict[int, int]:
    result: Dict[int, int] = {}
    for cluster_id in sorted(set(cluster_ids)):
        member_indices = [idx for idx, value in enumerate(cluster_ids) if value == cluster_id]
        member_tensor = space_matrix[member_indices]
        distances = torch.norm(member_tensor - centroids[cluster_id].unsqueeze(0), dim=1)
        result[cluster_id] = member_indices[int(torch.argmin(distances).item())]
    return result


def export_assignments(
    output_path: Path,
    charts: Sequence[str],
    raw_levels: Sequence[float],
    bucket_labels: Sequence[str],
    cluster_ids: Sequence[int],
    distances: Sequence[float],
    pca_coords: torch.Tensor,
    axis_scores: Dict[str, torch.Tensor],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    axis_names = list(axis_scores.keys())
    fieldnames = [
        "chart",
        "raw_level",
        "bucket_label",
        "cluster_id",
        "distance_to_centroid",
        "pca_x",
        "pca_y",
    ] + axis_names
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, chart in enumerate(charts):
            row = {
                "chart": chart,
                "raw_level": round_float(raw_levels[idx]),
                "bucket_label": bucket_labels[idx],
                "cluster_id": int(cluster_ids[idx]),
                "distance_to_centroid": round_float(distances[idx]),
                "pca_x": round_float(pca_coords[idx, 0].item()),
                "pca_y": round_float(pca_coords[idx, 1].item()),
            }
            for axis_name, values in axis_scores.items():
                row[axis_name] = round_float(values[idx].item())
            writer.writerow(row)


def export_cluster_summary(output_path: Path, rows: Sequence[Dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    axis_names = list(AXIS_METRIC_GROUPS.keys())
    fieldnames = [
        "cluster_id",
        "size",
        "raw_level_mean",
        "raw_level_min",
        "raw_level_max",
        "representative_chart",
        "label_distribution",
        "dominant_positive_axes",
        "dominant_negative_axes",
    ] + axis_names
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def export_cluster_members(output_dir: Path, members_by_cluster: Dict[int, List[Dict[str, object]]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for cluster_id, rows in members_by_cluster.items():
        if not rows:
            continue
        fieldnames = list(rows[0].keys())
        with (output_dir / f"cluster_{cluster_id}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster charts by interpretable style axes.")
    parser.add_argument("--features-csv", type=Path, default=Path("chart_style_features.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("cluster_style_analysis"))
    parser.add_argument("--num-clusters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cluster-space", choices=["axes", "full"], default="axes")
    args = parser.parse_args()

    parsed = parse_feature_matrix(load_rows(args.features_csv))
    metric_names = parsed["metric_names"]
    charts = parsed["charts"]
    bucket_labels = parsed["bucket_labels"]
    raw_levels = parsed["raw_levels"]
    matrix = parsed["matrix"]

    if args.num_clusters <= 1 or args.num_clusters > matrix.size(0):
        raise ValueError(f"--num-clusters must be in [2, {matrix.size(0)}], got {args.num_clusters}")

    scaler = StandardScaler()
    scaled_np = scaler.fit_transform(matrix.numpy())
    scaled_matrix = torch.tensor(scaled_np, dtype=torch.float32)

    cluster_input = build_cluster_input(metric_names, scaled_matrix, args.cluster_space)
    space_names = cluster_input["space_names"]
    space_matrix = cluster_input["space_matrix"]
    axis_scores = cluster_input["axis_scores"]

    model = KMeans(n_clusters=args.num_clusters, random_state=args.seed, n_init=20)
    cluster_ids = model.fit_predict(space_matrix.numpy()).tolist()
    centroids = torch.tensor(model.cluster_centers_, dtype=torch.float32)
    distances = [
        float(torch.norm(space_matrix[idx] - centroids[cluster_ids[idx]]).item())
        for idx in range(space_matrix.size(0))
    ]

    pca = PCA(n_components=2, random_state=args.seed)
    pca_coords = torch.tensor(pca.fit_transform(space_matrix.numpy()), dtype=torch.float32)
    silhouette = silhouette_score(space_matrix.numpy(), cluster_ids)

    assignments_path = args.output_dir / "cluster_assignments.csv"
    export_assignments(assignments_path, charts, raw_levels, bucket_labels, cluster_ids, distances, pca_coords, axis_scores)

    rep_indices = representative_indices(space_matrix, cluster_ids, centroids)
    summary_rows: List[Dict[str, object]] = []
    members_by_cluster: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    axis_names = list(AXIS_METRIC_GROUPS.keys())
    for cluster_id in sorted(set(cluster_ids)):
        member_indices = [idx for idx, value in enumerate(cluster_ids) if value == cluster_id]
        member_levels = [raw_levels[idx] for idx in member_indices]
        label_counts = Counter(bucket_labels[idx] for idx in member_indices)

        cluster_axis_means: Dict[str, float] = {}
        for axis_name in axis_names:
            cluster_axis_means[axis_name] = float(axis_scores[axis_name][member_indices].mean().item())

        sorted_positive = sorted(cluster_axis_means.items(), key=lambda item: item[1], reverse=True)
        sorted_negative = sorted(cluster_axis_means.items(), key=lambda item: item[1])
        row: Dict[str, object] = {
            "cluster_id": cluster_id,
            "size": len(member_indices),
            "raw_level_mean": round_float(sum(member_levels) / len(member_levels)),
            "raw_level_min": round_float(min(member_levels)),
            "raw_level_max": round_float(max(member_levels)),
            "representative_chart": charts[rep_indices[cluster_id]],
            "label_distribution": " ".join(
                f"{label}={label_counts.get(label, 0)}" for label in ["13", "13+", "14", "14+"]
            ),
            "dominant_positive_axes": ", ".join(
                f"{name}={round_float(value)}" for name, value in sorted_positive[:3]
            ),
            "dominant_negative_axes": ", ".join(
                f"{name}={round_float(value)}" for name, value in sorted_negative[:3]
            ),
        }
        for axis_name in axis_names:
            row[axis_name] = round_float(cluster_axis_means[axis_name])
        summary_rows.append(row)

        for idx in sorted(member_indices, key=lambda item: distances[item]):
            member_row: Dict[str, object] = {
                "chart": charts[idx],
                "raw_level": round_float(raw_levels[idx]),
                "bucket_label": bucket_labels[idx],
                "distance_to_centroid": round_float(distances[idx]),
                "pca_x": round_float(pca_coords[idx, 0].item()),
                "pca_y": round_float(pca_coords[idx, 1].item()),
            }
            for axis_name in axis_names:
                member_row[axis_name] = round_float(axis_scores[axis_name][idx].item())
            members_by_cluster[cluster_id].append(member_row)

    summary_path = args.output_dir / "cluster_summary.csv"
    export_cluster_summary(summary_path, summary_rows)
    export_cluster_members(args.output_dir / "cluster_members", members_by_cluster)

    print(f"num_charts={len(charts)}")
    print(f"cluster_space={args.cluster_space}")
    print(f"space_dim={space_matrix.size(1)}")
    print(f"num_clusters={args.num_clusters}")
    print(f"silhouette_score={round_float(silhouette)}")
    print(f"assignments_output={assignments_path}")
    print(f"summary_output={summary_path}")
    print(f"space_names={','.join(space_names)}")
    print("cluster_axis_summary")
    for row in summary_rows:
        print(
            f"cluster={row['cluster_id']} size={row['size']} "
            f"rep={row['representative_chart']} "
            f"pos=[{row['dominant_positive_axes']}] "
            f"neg=[{row['dominant_negative_axes']}]"
        )


if __name__ == "__main__":
    main()
