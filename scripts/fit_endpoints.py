"""Fit microtubule endpoints in a raw + label TIFF pair.

Usage:
    python fit_endpoints.py --raw raw.tif --labels labels.tif \\
        --sigma 1.6 1.6 --out endpoints.csv

Both inputs may be 2-D ``(H, W)`` or 2-D + time ``(T, H, W)`` TIFFs.
The CSV output schema matches ``kapoorlabs_mtrack.io.tif.ENDPOINT_CSV_COLUMNS``
and is consumed by ``track_endpoints.py`` in the next stage.
"""

from __future__ import annotations

import argparse
import sys

from kapoorlabs_mtrack.io import load_pair, save_endpoints_csv
from kapoorlabs_mtrack.pipeline.fit_stack import (
    fit_stack,
    snapshots_to_csv_rows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", required=True, help="raw image TIFF")
    p.add_argument("--labels", required=True, help="label image TIFF")
    p.add_argument(
        "--sigma",
        type=float,
        nargs=2,
        required=True,
        metavar=("SIGMA_X", "SIGMA_Y"),
        help="PSF widths in pixels (microscope property; not fitted).",
    )
    p.add_argument("--out", required=True, help="output CSV path")
    p.add_argument(
        "--ds-seed",
        type=float,
        default=0.7,
        help="initial curve step length (default 0.7)",
    )
    p.add_argument(
        "--pad",
        type=int,
        default=None,
        help="bbox padding in pixels (default: ceil(3 * max(sigma)))",
    )
    p.add_argument(
        "--jac-mode",
        choices=("analytic", "hybrid", "numeric"),
        default="hybrid",
        help="LM Jacobian mode (default hybrid for sub-pixel endpoints)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    raw, labels = load_pair(args.raw, args.labels)

    snapshots = fit_stack(
        raw.array,
        labels.array,
        sigma=tuple(args.sigma),
        pad=args.pad,
        ds_seed=args.ds_seed,
        jac_mode=args.jac_mode,
    )

    n_ok = sum(len(fs.mts) for fs in snapshots)
    n_skipped = sum(len(fs.skipped_labels) for fs in snapshots)
    print(
        f"frames={len(snapshots)}  mts_fit={n_ok}  labels_skipped={n_skipped}"
    )

    rows = snapshots_to_csv_rows(snapshots)
    save_endpoints_csv(args.out, rows)
    print(f"wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
