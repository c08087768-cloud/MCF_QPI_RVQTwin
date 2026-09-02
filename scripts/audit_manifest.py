#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from mcfqpi.config import save_json
from mcfqpi.data.audit import audit_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计 MCF-QPI manifest 的文件、尺寸与跨 split 泄漏")
    parser.add_argument("manifest")
    parser.add_argument("--output", default="outputs/data_inspection/manifest_audit.json")
    parser.add_argument("--skip-dimensions", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--strict-protocol", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_manifest(
        args.manifest,
        inspect_dimensions=not args.skip_dimensions,
        expected_rows=args.expected_rows,
        strict_protocol=args.strict_protocol,
    )
    save_json(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if args.fail_on_error and not report["passed_core_checks"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
