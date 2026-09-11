#!/usr/bin/env python
"""Audit, aggregate, and pair completed RBDP experiment directories."""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rbdp.io import read_json, write_csv, write_json  # noqa: E402
from rbdp.results import LONG_FIELDNAMES, PAIRED_FIELDNAMES  # noqa: E402
from rbdp.results import collect_runs, expand_expected_grids  # noqa: E402
from rbdp.results import missing_expected_runs, paired_comparisons  # noqa: E402
from rbdp.results import summarize_rows, summary_fieldnames  # noqa: E402


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", help="one or more result roots")
    parser.add_argument("--output-dir", required=True,
                        help="directory for audited CSV/JSON outputs")
    parser.add_argument("--baseline", default=None,
                        help="ablation name used for paired comparisons")
    parser.add_argument("--expected-grid", default=None,
                        help="optional JSON file containing a 'grids' list")
    parser.add_argument("--strict", action="store_true",
                        help="return non-zero when an invalid or expected run exists")
    return parser.parse_args()


def main():
    arguments = _arguments()
    rows, audit = collect_runs(arguments.roots)
    summaries = summarize_rows(rows)
    comparisons = []
    if arguments.baseline:
        comparisons = paired_comparisons(rows, arguments.baseline)

    expected, missing = [], []
    if arguments.expected_grid:
        expected = expand_expected_grids(read_json(arguments.expected_grid))
        missing = missing_expected_runs(rows, expected)

    output_dir = os.path.abspath(arguments.output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    write_csv(os.path.join(output_dir, "results.csv"), rows, LONG_FIELDNAMES)
    write_csv(os.path.join(output_dir, "summary.csv"), summaries,
              summary_fieldnames())
    write_csv(os.path.join(output_dir, "paired_comparisons.csv"), comparisons,
              PAIRED_FIELDNAMES)
    write_json(os.path.join(output_dir, "audit.json"), {
        "roots": [os.path.abspath(root) for root in arguments.roots],
        "valid_run_count": len(rows),
        "summary_group_count": len(summaries),
        "audit_record_count": len(audit),
        "invalid_record_count": sum(not record["valid"] for record in audit),
        "expected_cell_count": len(expected),
        "missing_expected_count": len(missing),
        "missing_expected_runs": missing,
        "records": audit,
    })
    print("valid_runs=%d summary_groups=%d invalid_records=%d missing_expected=%d" % (
        len(rows), len(summaries),
        sum(not record["valid"] for record in audit), len(missing)))
    print("wrote=%s" % output_dir)
    if arguments.strict and (missing or any(not record["valid"] for record in audit)):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
