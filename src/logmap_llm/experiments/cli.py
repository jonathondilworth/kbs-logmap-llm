"""Command-line interface for the minimal LogMapLLM batch harness."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

from logmap_llm.experiments.plan import generate_batch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="logmap-llm-batch",
        description="Generate, run, inspect, and aggregate local LogMapLLM batches.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="validate and freeze a batch TOML")
    generate.add_argument("spec", help="path to batch.toml")
    generate.add_argument(
        "--dry-run", action="store_true",
        help="validate and show the complete plan without writing files",
    )

    run = commands.add_parser("run", help="run selected jobs from a frozen batch")
    run.add_argument("batch", help="generated batch directory")
    run.add_argument("--jobs", type=int, default=None, help="maximum concurrent jobs")
    run.add_argument(
        "--resume", action="store_true",
        help="skip verified successes and retry other jobs in fresh attempts",
    )
    run.add_argument(
        "--select", action="append", default=[], metavar="KEY=A,B",
        help="select task/model/condition/repeat IDs; repeat for different keys",
    )
    run.add_argument("--limit", type=int, default=None, help="run at most N selected jobs")

    status = commands.add_parser("status", help="show manifest-derived batch status")
    status.add_argument("batch", help="generated batch directory")
    status.add_argument(
        "--select", action="append", default=[], metavar="KEY=A,B",
        help="show only matching jobs",
    )

    aggregate = commands.add_parser("aggregate", help="aggregate only this batch's jobs")
    aggregate.add_argument("batch", help="generated batch directory")
    aggregate.add_argument(
        "--allow-incomplete", action="store_true",
        help="compute explicitly marked summaries from incomplete groups",
    )

    import_command = commands.add_parser(
        "import-alignment",
        help="verify and import one precomputed initial alignment",
    )
    import_command.add_argument("batch", help="fresh generated batch directory")
    selector = import_command.add_mutually_exclusive_group(required=True)
    selector.add_argument(
        "--task", help="task whose sole alignment prerequisite is imported (preferred)",
    )
    selector.add_argument("--alignment-id", help="exact alignment prerequisite ID")
    import_command.add_argument(
        "--source-dir", required=True,
        help="directory containing the receipt-declared alignment files",
    )
    import_command.add_argument(
        "--receipt", required=True,
        help="schema-1 JSON transfer receipt",
    )
    import_command.add_argument(
        "--source-task-prefix",
        help="confirm the source filename prefix declared by the receipt",
    )
    return parser


def _result_code(result: Any) -> int:
    if isinstance(result, bool):
        return 0 if result else 1
    if isinstance(result, int):
        return result
    if isinstance(result, dict):
        value = result.get("exit_code", 0)
        return int(value) if isinstance(value, int) else 0
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the batch CLI and return a process exit code."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "generate":
            generate_batch(args.spec, dry_run=args.dry_run)
            return 0

        if args.command == "run":
            from logmap_llm.experiments.run import run_batch

            result = run_batch(
                args.batch,
                jobs_override=args.jobs,
                resume=args.resume,
                selectors=tuple(args.select),
                limit=args.limit,
            )
            return _result_code(result)

        if args.command == "status":
            from logmap_llm.experiments.run import print_status

            result = print_status(args.batch, selectors=tuple(args.select))
            return _result_code(result)

        if args.command == "aggregate":
            from logmap_llm.experiments.aggregate import aggregate_batch

            result = aggregate_batch(args.batch, allow_incomplete=args.allow_incomplete)
            return _result_code(result)

        if args.command == "import-alignment":
            from logmap_llm.experiments.import_alignment import import_alignment

            result = import_alignment(
                args.batch,
                task=args.task,
                alignment_id=args.alignment_id,
                source_dir=args.source_dir,
                receipt_path=args.receipt,
                source_task_prefix=args.source_task_prefix,
            )
            print(
                f"Imported alignment {result['id']} "
                f"({len(result['artifacts'])} verified artifacts)."
            )
            return 0

        raise RuntimeError(f"unhandled command: {args.command}")
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
