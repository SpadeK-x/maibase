import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def format_label_distribution(label_names: Sequence[str], counts: Counter) -> str:
    parts = [f"{label}={counts.get(label, 0)}" for label in label_names]
    return " ".join(parts)


def round_float(value: float) -> float:
    return round(float(value), 6)


def load_embedding_payload(path: Path) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu")
    required_keys = {"charts", "bucket_labels", "bucket_label_names", "raw_levels", "embeddings", "feature_names"}
    missing = required_keys.difference(payload.keys())
    if missing:
        raise KeyError(f"Embedding payload missing keys: {sorted(missing)}")
    return payload


def representative_indices(
    standardized_embeddings: torch.Tensor,
    cluster_ids: Sequence[int],
    centroids: torch.Tensor,
) -> Dict[int, int]:
    reps: Dict[int, int] = {}
    for cluster_id in sorted(set(cluster_ids)):
        member_indices = [idx for idx, value in enumerate(cluster_ids) if value == cluster_id]
        member_tensor = standardized_embeddings[member_indices]
        centroid = centroids[cluster_id].unsqueeze(0)
        distances = torch.norm(member_tensor - centroid, dim=1)
        best_local = int(torch.argmin(distances).item())
        reps[cluster_id] = member_indices[best_local]
    return reps


def top_feature_shifts(
    cluster_mean: torch.Tensor,
    global_mean: torch.Tensor,
    feature_names: Sequence[str],
    top_k: int,
) -> Tuple[str, str]:
    delta = cluster_mean - global_mean
    top_pos_idx = torch.topk(delta, k=min(top_k, delta.numel())).indices.tolist()
    top_neg_idx = torch.topk(-delta, k=min(top_k, delta.numel())).indices.tolist()
    top_pos = ", ".join(f"{feature_names[idx]}={round_float(delta[idx].item())}" for idx in top_pos_idx)
    top_neg = ", ".join(f"{feature_names[idx]}={round_float(delta[idx].item())}" for idx in top_neg_idx)
    return top_pos, top_neg


def export_assignment_csv(
    output_path: Path,
    charts: Sequence[str],
    raw_levels: Sequence[float],
    bucket_labels: Sequence[str],
    cluster_ids: Sequence[int],
    distances: Sequence[float],
    pca_coords: torch.Tensor,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "chart",
                "raw_level",
                "bucket_label",
                "cluster_id",
                "distance_to_centroid",
                "pca_x",
                "pca_y",
            ],
        )
        writer.writeheader()
        for idx, chart in enumerate(charts):
            writer.writerow(
                {
                    "chart": chart,
                    "raw_level": round_float(raw_levels[idx]),
                    "bucket_label": bucket_labels[idx],
                    "cluster_id": int(cluster_ids[idx]),
                    "distance_to_centroid": round_float(distances[idx]),
                    "pca_x": round_float(pca_coords[idx, 0].item()),
                    "pca_y": round_float(pca_coords[idx, 1].item()),
                }
            )


def export_cluster_summary_csv(
    output_path: Path,
    rows: Sequence[Dict[str, object]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cluster_id",
        "size",
        "raw_level_mean",
        "raw_level_min",
        "raw_level_max",
        "representative_chart",
        "label_distribution",
        "top_positive_features",
        "top_negative_features",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_cluster_members(output_dir: Path, members_by_cluster: Dict[int, List[Dict[str, object]]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for cluster_id, rows in members_by_cluster.items():
        output_path = output_dir / f"cluster_{cluster_id}.csv"
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["chart", "raw_level", "bucket_label", "distance_to_centroid", "pca_x", "pca_y"],
            )
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster chart-level baseline embeddings and export cluster analysis.")
    parser.add_argument("--embeddings-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("cluster_analysis"))
    parser.add_argument("--num-clusters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k-features", type=int, default=8)
    args = parser.parse_args()

    payload = load_embedding_payload(args.embeddings_path)
    charts = [str(value) for value in payload["charts"]]
    feature_names = [str(value) for value in payload["feature_names"]]
    raw_levels_tensor = torch.as_tensor(payload["raw_levels"], dtype=torch.float32)
    raw_levels = [float(value) for value in raw_levels_tensor.tolist()]
    label_names = [str(value) for value in payload["bucket_label_names"]]
    label_indices = torch.as_tensor(payload["bucket_labels"], dtype=torch.long).tolist()
    bucket_labels = [label_names[int(index)] for index in label_indices]
    embeddings = torch.as_tensor(payload["embeddings"], dtype=torch.float32)

    if embeddings.dim() != 2:
        raise ValueError(f"Expected embeddings shape [N, D], got {tuple(embeddings.shape)}")
    if args.num_clusters <= 1 or args.num_clusters > embeddings.size(0):
        raise ValueError(
            f"--num-clusters must be in [2, {embeddings.size(0)}], got {args.num_clusters}"
        )

    scaler = StandardScaler()
    scaled_np = scaler.fit_transform(embeddings.numpy())
    scaled = torch.tensor(scaled_np, dtype=torch.float32)

    model = KMeans(n_clusters=args.num_clusters, random_state=args.seed, n_init=20)
    cluster_ids = model.fit_predict(scaled_np).tolist()
    centroid_np = model.cluster_centers_
    centroids = torch.tensor(centroid_np, dtype=torch.float32)

    distances: List[float] = []
    for idx, cluster_id in enumerate(cluster_ids):
        distance = torch.norm(scaled[idx] - centroids[cluster_id]).item()
        distances.append(float(distance))

    pca = PCA(n_components=2, random_state=args.seed)
    pca_coords = torch.tensor(pca.fit_transform(scaled_np), dtype=torch.float32)
    silhouette = silhouette_score(scaled_np, cluster_ids)

    assignment_path = args.output_dir / "cluster_assignments.csv"
    export_assignment_csv(
        assignment_path,
        charts,
        raw_levels,
        bucket_labels,
        cluster_ids,
        distances,
        pca_coords,
    )

    global_mean = scaled.mean(dim=0)
    rep_index_map = representative_indices(scaled, cluster_ids, centroids)

    summary_rows: List[Dict[str, object]] = []
    members_by_cluster: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for cluster_id in sorted(set(cluster_ids)):
        member_indices = [idx for idx, value in enumerate(cluster_ids) if value == cluster_id]
        member_levels = [raw_levels[idx] for idx in member_indices]
        member_label_counts = Counter(bucket_labels[idx] for idx in member_indices)
        member_scaled = scaled[member_indices]
        cluster_mean = member_scaled.mean(dim=0)
        top_pos, top_neg = top_feature_shifts(cluster_mean, global_mean, feature_names, args.top_k_features)
        rep_idx = rep_index_map[cluster_id]

        summary_rows.append(
            {
                "cluster_id": cluster_id,
                "size": len(member_indices),
                "raw_level_mean": round_float(sum(member_levels) / len(member_levels)),
                "raw_level_min": round_float(min(member_levels)),
                "raw_level_max": round_float(max(member_levels)),
                "representative_chart": charts[rep_idx],
                "label_distribution": format_label_distribution(label_names, member_label_counts),
                "top_positive_features": top_pos,
                "top_negative_features": top_neg,
            }
        )

        sorted_members = sorted(member_indices, key=lambda idx: distances[idx])
        for idx in sorted_members:
            members_by_cluster[cluster_id].append(
                {
                    "chart": charts[idx],
                    "raw_level": round_float(raw_levels[idx]),
                    "bucket_label": bucket_labels[idx],
                    "distance_to_centroid": round_float(distances[idx]),
                    "pca_x": round_float(pca_coords[idx, 0].item()),
                    "pca_y": round_float(pca_coords[idx, 1].item()),
                }
            )

    summary_path = args.output_dir / "cluster_summary.csv"
    export_cluster_summary_csv(summary_path, summary_rows)
    export_cluster_members(args.output_dir / "cluster_members", members_by_cluster)

    print(f"num_charts={len(charts)}")
    print(f"embedding_dim={embeddings.size(1)}")
    print(f"num_clusters={args.num_clusters}")
    print(f"silhouette_score={round_float(silhouette)}")
    print(f"assignments_output={assignment_path}")
    print(f"summary_output={summary_path}")
    print(f"members_dir={args.output_dir / 'cluster_members'}")
    print("cluster_label_distributions")
    for row in summary_rows:
        print(
            f"cluster={row['cluster_id']} size={row['size']} "
            f"raw_level_mean={row['raw_level_mean']} "
            f"{row['label_distribution']}"
        )


if __name__ == "__main__":
    main()
