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


def round_float(value: float) -> float:
    return round(float(value), 6)


def load_embedding_payload(path: Path) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu")
    required = {"charts", "bucket_labels", "embeddings"}
    missing = required.difference(payload.keys())
    if missing:
        raise KeyError(f"Embedding payload missing keys: {sorted(missing)}")
    return payload


def representative_indices(
    scaled_embeddings: torch.Tensor,
    cluster_ids: Sequence[int],
    centroids: torch.Tensor,
) -> Dict[int, int]:
    reps: Dict[int, int] = {}
    for cluster_id in sorted(set(cluster_ids)):
        member_indices = [idx for idx, value in enumerate(cluster_ids) if value == cluster_id]
        member_tensor = scaled_embeddings[member_indices]
        centroid = centroids[cluster_id].unsqueeze(0)
        distances = torch.norm(member_tensor - centroid, dim=1)
        reps[cluster_id] = member_indices[int(torch.argmin(distances).item())]
    return reps


def export_csv(output_path: Path, rows: Sequence[Dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster chart embeddings exported from masked event modeling.")
    parser.add_argument("--embeddings-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("masked_embedding_clusters"))
    parser.add_argument("--num-clusters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    payload = load_embedding_payload(args.embeddings_path)
    charts = [str(value) for value in payload["charts"]]
    embeddings = torch.as_tensor(payload["embeddings"], dtype=torch.float32)
    label_names = [str(value) for value in payload.get("bucket_label_names", ["13", "13+", "14", "14+"])]
    bucket_indices = torch.as_tensor(payload["bucket_labels"], dtype=torch.long).tolist()
    bucket_labels = [label_names[idx] if 0 <= int(idx) < len(label_names) else "unknown" for idx in bucket_indices]
    raw_levels = [float(value) for value in torch.as_tensor(payload.get("raw_levels", torch.zeros(len(charts))), dtype=torch.float32).tolist()]

    if embeddings.dim() != 2:
        raise ValueError(f"Expected embeddings shape [N, D], got {tuple(embeddings.shape)}")
    if args.num_clusters <= 1 or args.num_clusters > embeddings.size(0):
        raise ValueError(f"--num-clusters must be in [2, {embeddings.size(0)}], got {args.num_clusters}")

    scaler = StandardScaler()
    scaled_np = scaler.fit_transform(embeddings.numpy())
    scaled = torch.tensor(scaled_np, dtype=torch.float32)

    model = KMeans(n_clusters=args.num_clusters, random_state=args.seed, n_init=20)
    cluster_ids = model.fit_predict(scaled_np).tolist()
    centroids = torch.tensor(model.cluster_centers_, dtype=torch.float32)
    distances = [float(torch.norm(scaled[idx] - centroids[cluster_ids[idx]]).item()) for idx in range(scaled.size(0))]

    pca = PCA(n_components=2, random_state=args.seed)
    pca_coords = torch.tensor(pca.fit_transform(scaled_np), dtype=torch.float32)
    silhouette = silhouette_score(scaled_np, cluster_ids)

    assignment_rows: List[Dict[str, object]] = []
    for idx, chart in enumerate(charts):
        assignment_rows.append(
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
    export_csv(args.output_dir / "cluster_assignments.csv", assignment_rows)

    rep_indices = representative_indices(scaled, cluster_ids, centroids)
    summary_rows: List[Dict[str, object]] = []
    members_by_cluster: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for cluster_id in sorted(set(cluster_ids)):
        member_indices = [idx for idx, value in enumerate(cluster_ids) if value == cluster_id]
        level_values = [raw_levels[idx] for idx in member_indices]
        label_counts = Counter(bucket_labels[idx] for idx in member_indices)
        summary_rows.append(
            {
                "cluster_id": cluster_id,
                "size": len(member_indices),
                "raw_level_mean": round_float(sum(level_values) / len(level_values)),
                "raw_level_min": round_float(min(level_values)),
                "raw_level_max": round_float(max(level_values)),
                "representative_chart": charts[rep_indices[cluster_id]],
                "label_distribution": " ".join(f"{label}={label_counts.get(label, 0)}" for label in label_names),
            }
        )
        for idx in sorted(member_indices, key=lambda item: distances[item]):
            members_by_cluster[cluster_id].append(assignment_rows[idx])

    export_csv(args.output_dir / "cluster_summary.csv", summary_rows)
    members_dir = args.output_dir / "cluster_members"
    for cluster_id, rows in members_by_cluster.items():
        export_csv(members_dir / f"cluster_{cluster_id}.csv", rows)

    print(f"num_charts={len(charts)}")
    print(f"embedding_dim={embeddings.size(1)}")
    print(f"num_clusters={args.num_clusters}")
    print(f"silhouette_score={round_float(silhouette)}")
    print(f"assignments_output={args.output_dir / 'cluster_assignments.csv'}")
    print(f"summary_output={args.output_dir / 'cluster_summary.csv'}")
    for row in summary_rows:
        print(
            f"cluster={row['cluster_id']} size={row['size']} "
            f"raw_level_mean={row['raw_level_mean']} "
            f"{row['label_distribution']}"
        )


if __name__ == "__main__":
    main()
