"""Command line: ``python -m mlp_data list|fetch [name ...]|where``."""
import sys

from . import DATASETS, data_dir


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "list"
    if cmd == "where":
        print(data_dir())
    elif cmd == "list":
        print("Datasets:", ", ".join(DATASETS))
        print("Fetch with: python -m mlp_data fetch [name ...]   (no name = all)")
    elif cmd == "fetch":
        names = argv[1:] or list(DATASETS)
        unknown = [n for n in names if n not in DATASETS]
        if unknown:
            print(f"Unknown dataset(s): {unknown}. Choose from {list(DATASETS)}.", file=sys.stderr)
            return 2
        for n in names:
            obj = DATASETS[n]()
            shape = getattr(obj, "shape", None) or {k: v.shape for k, v in obj._asdict().items()}
            print(f"[ok] {n}: {shape}")
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
