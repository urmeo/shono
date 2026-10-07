"""Check notebook syntax and cleared outputs without running training or APIs."""

import argparse
import json
import sys
from pathlib import Path

from IPython.core.inputtransformer2 import TransformerManager


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.parse_args()
    transformer = TransformerManager()
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "notebooks").glob("*.ipynb"))
    if len(paths) != 4:
        raise ValueError("expected four notebook drivers")
    count = 0
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(document["cells"]):
            if cell["cell_type"] == "code":
                if cell.get("outputs") or cell.get("execution_count") is not None:
                    raise ValueError(f"{path.name} cell{index}: saved output must be cleared")
                source = "".join(cell["source"])
                compile(
                    transformer.transform_cell(source),
                    f"{path.name}:cell{index}",
                    "exec",
                )
                count += 1
    print(f"{len(paths)} notebooks; {count} code cells compile; no saved outputs")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
