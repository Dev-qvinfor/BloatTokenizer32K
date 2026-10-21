"""Command-line tools for counting tokens with the Bloat Tokenizer."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from .tokenizer import CurriculumTokenizer


TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".py", ".pyi", ".rs", ".c", ".h", ".cc",
    ".cpp", ".hpp", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".kts",
    ".go", ".cs", ".json", ".jsonl", ".ndjson", ".csv", ".tsv", ".html",
    ".htm", ".css", ".scss", ".sql", ".xml", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".conf", ".tex", ".latex", ".apl", ".pl", ".pm",
}
FORMAT_BY_SUFFIX = {
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".csv": "csv",
    ".tsv": "tsv",
    ".parquet": "parquet",
}


def _format_for(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    return FORMAT_BY_SUFFIX.get(path.suffix.lower(), "text")


def _field_value(value: Any, field: str) -> Any:
    current = value
    for part in field.split("."):
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(f"field '{field}' is missing at '{part}'")
            current = current[part]
        elif isinstance(current, list) and part.isdigit():
            current = current[int(part)]
        else:
            raise KeyError(f"field '{field}' is not available in this record")
    return current


def _as_text(value: Any) -> Iterator[str]:
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _as_text(item)
        return
    if isinstance(value, dict):
        yield json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return
    if isinstance(value, bytes):
        yield value.decode("utf-8", errors="replace")
        return
    yield str(value)


def _text_records(path: Path) -> Iterator[str]:
    # A whole-file read preserves exact BPE boundaries for ordinary text files.
    yield path.read_text(encoding="utf-8")


def _json_records(path: Path, field: Optional[str]) -> Iterator[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if field:
        yield from _as_text(_field_value(payload, field))
    else:
        yield path.read_text(encoding="utf-8")


def _jsonl_records(path: Path, field: Optional[str]) -> Iterator[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if field:
                try:
                    yield from _as_text(_field_value(json.loads(line), field))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{line_number}: cannot read field '{field}': {exc}") from exc
            else:
                yield line


def _delimited_records(path: Path, field: Optional[str], delimiter: str) -> Iterator[str]:
    if not field:
        with path.open("r", encoding="utf-8", newline="") as handle:
            yield from handle
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames or field not in reader.fieldnames:
            available = ", ".join(reader.fieldnames or [])
            raise ValueError(f"{path}: field '{field}' not found; available columns: {available}")
        for row in reader:
            yield from _as_text(row.get(field))


def _parquet_records(path: Path, field: Optional[str]) -> Iterator[str]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet support requires pyarrow: pip install pyarrow") from exc

    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    if field:
        if field not in schema.names:
            raise ValueError(f"{path}: field '{field}' not found; available columns: {', '.join(schema.names)}")
        columns = [field]
    else:
        columns = [
            name for name, column in zip(schema.names, schema)
            if pa.types.is_string(column.type)
            or pa.types.is_large_string(column.type)
            or pa.types.is_binary(column.type)
            or pa.types.is_large_binary(column.type)
        ]
        if not columns:
            raise ValueError(f"{path}: no string or binary columns; pass --field to select a column")

    for batch in parquet.iter_batches(batch_size=1024, columns=columns):
        data = batch.to_pydict()
        for column in columns:
            for value in data[column]:
                yield from _as_text(value)


def _records(path: Path, file_format: str, field: Optional[str]) -> Iterator[str]:
    if file_format == "text":
        yield from _text_records(path)
    elif file_format == "json":
        yield from _json_records(path, field)
    elif file_format == "jsonl":
        yield from _jsonl_records(path, field)
    elif file_format == "csv":
        yield from _delimited_records(path, field, ",")
    elif file_format == "tsv":
        yield from _delimited_records(path, field, "\t")
    elif file_format == "parquet":
        yield from _parquet_records(path, field)
    else:
        raise ValueError(f"Unsupported input format: {file_format}")


def _input_files(input_path: Path, recursive: bool) -> tuple[list[Path], list[str]]:
    if input_path.is_file():
        return [input_path], []
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    candidates = input_path.rglob("*") if recursive else input_path.glob("*")
    files: list[Path] = []
    skipped: list[str] = []
    for path in sorted(candidates):
        if not path.is_file():
            continue
        if path.suffix.lower() in TEXT_EXTENSIONS or path.suffix.lower() == ".parquet":
            files.append(path)
        else:
            skipped.append(str(path))
    return files, skipped


def _count_file(
    tokenizer: CurriculumTokenizer,
    path: Path,
    file_format: str,
    field: Optional[str],
    add_special_tokens: bool,
    batch_size: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    records = 0
    characters = 0
    utf8_bytes = 0
    tokens = 0
    batch: list[str] = []

    def flush() -> None:
        nonlocal tokens
        if not batch:
            return
        encoded = tokenizer.encode_batch(batch, add_special_tokens=add_special_tokens)
        tokens += sum(len(ids) for ids in encoded["input_ids"])
        batch.clear()

    for text in _records(path, file_format, field):
        records += 1
        characters += len(text)
        utf8_bytes += len(text.encode("utf-8"))
        batch.append(text)
        if len(batch) >= batch_size:
            flush()
    flush()
    elapsed = time.perf_counter() - started
    return {
        "path": str(path),
        "format": file_format,
        "field": field,
        "records": records,
        "characters": characters,
        "utf8_bytes": utf8_bytes,
        "tokens": tokens,
        "chars_per_token": characters / tokens if tokens else 0.0,
        "seconds": elapsed,
        "tokens_per_second": tokens / elapsed if elapsed else 0.0,
    }


def _sum_stats(files: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("records", "characters", "utf8_bytes", "tokens")
    totals = {key: sum(item[key] for item in files) for key in keys}
    totals["chars_per_token"] = totals["characters"] / totals["tokens"] if totals["tokens"] else 0.0
    totals["seconds"] = sum(item["seconds"] for item in files)
    totals["tokens_per_second"] = totals["tokens"] / totals["seconds"] if totals["seconds"] else 0.0
    return totals


def _print_report(report: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    total = report["totals"]
    print(f"Tokenizer: {report['tokenizer']} (vocab {report['vocab_size']:,})")
    print(f"Input: {report['input']}")
    print(f"Files counted: {len(report['files'])} | skipped: {len(report['skipped'])}")
    print(f"Records: {total['records']:,}")
    print(f"Characters: {total['characters']:,} | UTF-8 bytes: {total['utf8_bytes']:,}")
    print(f"Tokens: {total['tokens']:,} | chars/token: {total['chars_per_token']:.3f}")
    print(f"Speed: {total['tokens_per_second']:,.0f} tokens/sec")
    for item in report["files"]:
        print(f"  {item['path']}: {item['tokens']:,} tokens ({item['format']})")


def count_command(args: argparse.Namespace) -> int:
    input_path = args.input.resolve()
    files, skipped = _input_files(input_path, args.recursive)
    if not files:
        raise ValueError(f"No supported input files found in {input_path}")
    tokenizer = CurriculumTokenizer.from_pretrained(str(args.artifacts.resolve()))
    counted: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in files:
        try:
            counted.append(_count_file(
                tokenizer=tokenizer,
                path=path,
                file_format=_format_for(path, args.format),
                field=args.field,
                add_special_tokens=args.add_special_tokens,
                batch_size=args.batch_size,
            ))
        except Exception as exc:
            errors.append({"path": str(path), "error": str(exc)})
            if args.fail_fast:
                raise
    if not counted:
        raise RuntimeError("No files could be counted successfully")
    report = {
        "tokenizer": "bloat-tokenizer",
        "vocab_size": tokenizer.vocab_size,
        "input": str(input_path),
        "field": args.field,
        "add_special_tokens": args.add_special_tokens,
        "files": counted,
        "skipped": skipped,
        "errors": errors,
        "totals": _sum_stats(counted),
    }
    if args.report:
        report_path = args.report.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_report(report, args.json)
    return 1 if errors else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bloat-tokenizer", description="Use Bloat Tokenizer artifacts from other projects.")
    parser.add_argument("--version", action="version", version="bloat-tokenizer 0.3.0")
    commands = parser.add_subparsers(dest="command", required=True)
    count = commands.add_parser("count", help="Count tokens in a file or directory")
    count.add_argument("--artifacts", type=Path, required=True, help="Directory containing vocab.json and merges.txt")
    count.add_argument("--input", type=Path, required=True, help="Input file or directory")
    count.add_argument("--format", choices=("auto", "text", "json", "jsonl", "csv", "tsv", "parquet"), default="auto")
    count.add_argument("--field", help="Text field/column for JSON, JSONL, CSV/TSV, or Parquet")
    count.add_argument("--recursive", action="store_true", help="Recurse through input directories")
    count.add_argument("--batch-size", type=int, default=256, help="Records encoded per batch (default: 256)")
    count.add_argument("--add-special-tokens", action="store_true", help="Append EOS to each counted record")
    count.add_argument("--fail-fast", action="store_true", help="Stop on the first unreadable file")
    count.add_argument("--json", action="store_true", help="Print the full JSON report")
    count.add_argument("--report", type=Path, help="Also save the JSON report to this path")
    count.set_defaults(handler=count_command)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    try:
        return args.handler(args)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(f"bloat-tokenizer: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
