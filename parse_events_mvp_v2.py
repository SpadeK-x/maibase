import argparse
import json
from pathlib import Path
from typing import Dict

from parse_events_mvp import (
    SLIDE_SHAPE_GROUP_MAP,
    SLIDE_SHAPE_SPAN_MAP,
    SimaiMVPParser,
    extract_section_lines,
)


class SimaiMVPParserV2(SimaiMVPParser):
    def slide_shape_group(self, slide):
        if slide is None:
            return "none"
        return SLIDE_SHAPE_GROUP_MAP.get(slide.raw_shape, "none")

    def slide_span(self, slide):
        return super().slide_span(slide)

    def generate_records(self, event_groups):
        records = super().generate_records(event_groups)
        for record in records:
            if "slide_direction" in record:
                del record["slide_direction"]
        return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Simai maidata into revised MVP 21-field event records.")
    parser.add_argument("maidata", type=Path, help="Path to maidata.txt")
    parser.add_argument("--difficulty", default="6", choices=["4", "5", "6"], help="Chart difficulty section")
    parser.add_argument("--output", type=Path, help="Optional output JSON path")
    args = parser.parse_args()

    section_lines = extract_section_lines(args.maidata, args.difficulty)
    parser_impl = SimaiMVPParserV2()
    records = parser_impl.parse_chart_section(section_lines)

    text = json.dumps(records, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
