import csv
from collections.abc import Iterator
from pathlib import Path


def delimited_rows(
    path: str | Path, *, delimiter: str = "\t", encoding: str = "utf-8-sig"
) -> Iterator[tuple[int, list[str]]]:
    with Path(path).open(encoding=encoding, newline="") as stream:
        reader = csv.reader(stream, delimiter=delimiter, strict=True)
        try:
            for values in reader:
                yield reader.line_num, values
        except csv.Error as error:
            kind = "CSV" if delimiter == "," else "TSV"
            raise ValueError(
                f"{path}: invalid {kind} on line {reader.line_num}: {error}"
            ) from error


def named_rows(
    path: str | Path,
    required: set[str],
    *,
    delimiter: str = "\t",
    encoding: str = "utf-8-sig",
    skip_blank: bool = False,
) -> Iterator[tuple[int, dict[str, str]]]:
    rows = delimited_rows(path, delimiter=delimiter, encoding=encoding)
    _, header = next(rows, (0, []))
    if len(header) != len(set(header)):
        raise ValueError(f"{path}: duplicate headers")
    if not header or not required.issubset(header):
        raise ValueError(f"{path}: required columns {sorted(required)}")
    for lineno, values in rows:
        if skip_blank and not values:
            continue
        if len(values) != len(header):
            raise ValueError(
                f"{path}: line {lineno}: expected {len(header)} columns, got {len(values)}"
            )
        yield lineno, dict(zip(header, values, strict=True))
