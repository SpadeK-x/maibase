import argparse
import json
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from parse_events_mvp_v2 import SimaiMVPParserV2, extract_section_lines


DIFFICULTY_NAME_MAP = {
    "4": "expert",
    "5": "master",
    "6": "remaster",
}


def discover_chart_files(charts_root: Path) -> List[Path]:
    return sorted(path for path in charts_root.glob("*/*/*.txt") if path.is_file())


def build_output_name(chart_file: Path, difficulty: str) -> str:
    parent_name = chart_file.parent.name
    chart_id = parent_name.split("_")[0]
    suffix = DIFFICULTY_NAME_MAP[difficulty]
    return f"{chart_id}_{suffix}.json"


def process_chart_file(
    parser_impl: SimaiMVPParserV2,
    chart_file: Path,
    difficulty: str,
) -> List[Dict[str, object]]:
    lines = extract_section_lines(chart_file, difficulty)
    if not lines:
        return []
    return parser_impl.parse_chart_section(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-convert Simai chart txt files into event-json files.")
    parser.add_argument(
        "--charts-root",
        type=Path,
        default=Path("charts"),
        help="Root directory containing chart txt files under charts/*/*/*.txt",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="6",
        choices=["4", "5", "6"],
        help="Difficulty section to export",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write per-chart event JSON files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit on number of chart files to process",
    )
    args = parser.parse_args()

    charts_root = args.charts_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    chart_files = discover_chart_files(charts_root)
    if args.limit is not None:
        chart_files = chart_files[: args.limit]

    parser_impl = SimaiMVPParserV2()
    written = 0
    skipped = 0

    for chart_file in tqdm(chart_files):
        try:
            records = process_chart_file(parser_impl, chart_file, args.difficulty)
            if not records:
                skipped += 1
                continue

            output_name = build_output_name(chart_file, args.difficulty)
            output_path = output_dir / output_name
            output_path.write_text(
                json.dumps(records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            written += 1
        except Exception as exc:
            skipped += 1
            print(f"skip {chart_file}: {exc}")

    print(f"written={written}")
    print(f"skipped={skipped}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
