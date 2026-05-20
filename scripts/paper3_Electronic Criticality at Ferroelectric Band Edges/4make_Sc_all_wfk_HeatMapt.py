#!/usr/bin/env python3
"""
Direct gap heatmap by k-point — all concentrations and ±P.

x-axis = k-point
y-axis = concentration / polarization
color  = direct gap in eV

Example:
python3 make_Sc_gap_heatmap.py \
  --label "Sc–HfO2" \
  --dataset "3.125%:/path/minus_WFK.nc:/path/plus_WFK.nc" \
  --dataset "6.25%:/path/minus_WFK.nc:/path/plus_WFK.nc" \
  --dataset "9.375%:/path/minus_WFK.nc:/path/plus_WFK.nc" \
  --out Sc_gap_heatmap.png
=================================================
python3 4make_Sc_all_wfk_HeatMapt.py \
  --label "Sc–HfO2" \
  --dataset "3.125%:/home/yanan/Material/Y_dope_project/3.125Per-Sc-HfO2-2x2x2/berry-scf-Results/berry_MinusP_3.125Sc_HfO2_12144332/berry_MinusP_3o_WFK.nc:/home/yanan/Material/Y_dope_project/3.125Per-Sc-HfO2-2x2x2/berry-scf-Results/berry_PlusP_3.125Sc_HfO2_12144296/berry_PlusP_3o_WFK.nc" \
  --dataset "6.25%:/home/yanan/Material/Y_dope_project/6.25Per-Sc-HfO2-2x2x2/berry-scf-results/berry_MinusP_6.25Sc_HfO2_12144457/berry_MinusP_6o_WFK.nc:/home/yanan/Material/Y_dope_project/6.25Per-Sc-HfO2-2x2x2/berry-scf-results/berry_PlusP_6.25Sc_HfO2_12144349/berry_PlusP_6o_WFK.nc" \
  --dataset "9.375%:/home/yanan/Material/Y_dope_project/9.375Sc-HfO2-2x2x2/berry-scf-Results/berry_MinusP_9.375Sc_HfO2_12144529/berry_MinusP_9o_WFK.nc:/home/yanan/Material/Y_dope_project/9.375Sc-HfO2-2x2x2/berry-scf-Results/berry_PlusP_9.375Sc_HfO2_12144507/berry_PlusP_9o_WFK.nc" \
  --out 4Sc_direct_gap_heatmap.png \
  --annotate
===================================================
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

Ha_to_eV = 27.211386245988


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\nSTDERR:\n{p.stderr}")
    return p.stdout


def ensure_small_nc(wfk_path: Path, small_path: Path):
    if small_path.exists() and small_path.stat().st_size > 0:
        return

    vars_keep = (
        "reduced_coordinates_of_kpoints,"
        "eigenvalues,occupations"
    )
    run(["ncks", "-O", "-v", vars_keep, str(wfk_path), str(small_path)])


def kpt_label_2x2x2(k):
    kk = tuple(int(round(2 * x)) for x in k)
    mapping = {
        (0, 0, 0): r"$\Gamma$",
        (1, 0, 0): "X",
        (0, 1, 0): "Y",
        (1, 1, 0): "S",
        (0, 0, 1): "Z",
        (1, 0, 1): "U",
        (0, 1, 1): "T",
        (1, 1, 1): "R",
    }
    return mapping.get(kk, f"k={k}")


def compute_gapk_from_small(small_nc: Path, occ_occ=1.5, occ_emp=0.5):
    f = Dataset(str(small_nc), "r")
    kpts = np.array(f.variables["reduced_coordinates_of_kpoints"][:])
    e = np.array(f.variables["eigenvalues"][0, :, :])
    occ = np.array(f.variables["occupations"][0, :, :])
    f.close()

    nk, nb = e.shape
    gapk = np.full(nk, np.nan)

    for ik in range(nk):
        occ_idx = np.where(occ[ik] > occ_occ)[0]
        unocc_idx = np.where(occ[ik] < occ_emp)[0]

        if len(occ_idx) == 0 or len(unocc_idx) == 0:
            continue

        homo = np.max(e[ik, occ_idx])
        lumo = np.min(e[ik, unocc_idx])
        gapk[ik] = lumo - homo

    return kpts, gapk


def load_gapk_from_wfk(wfk_path: Path, occ_occ=1.5, occ_emp=0.5):
    wfk_path = wfk_path.expanduser().resolve()
    if not wfk_path.exists():
        raise FileNotFoundError(f"Cannot find WFK: {wfk_path}")

    small_nc = wfk_path.with_name(wfk_path.stem + "_small_bandinfo.nc")
    ensure_small_nc(wfk_path, small_nc)
    return compute_gapk_from_small(small_nc, occ_occ, occ_emp)


def assert_same_kmesh(kref, ktest, tol=1e-8):
    if kref.shape != ktest.shape:
        raise ValueError("Different number of k-points.")
    if not np.allclose(kref, ktest, atol=tol, rtol=0):
        raise ValueError("k-point order mismatch.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True,
                    help='Format: "LABEL:minus_WFK.nc:plus_WFK.nc"')
    ap.add_argument("--label", default="Doped HfO2")
    ap.add_argument("--out", default="gap_heatmap.png")
    ap.add_argument("--occ_occ", type=float, default=1.5)
    ap.add_argument("--occ_emp", type=float, default=0.5)
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--annotate", action="store_true",
                    help="Write numerical gap values inside heatmap cells.")
    args = ap.parse_args()

    rows = []
    row_labels = []
    kref = None
    klabels = None

    for ds in args.dataset:
        parts = ds.split(":")
        if len(parts) != 3:
            raise ValueError(f'Bad dataset: {ds}')

        conc = parts[0].strip()
        minus_path = Path(parts[1].strip())
        plus_path = Path(parts[2].strip())

        k_m, gap_m = load_gapk_from_wfk(minus_path, args.occ_occ, args.occ_emp)
        k_p, gap_p = load_gapk_from_wfk(plus_path, args.occ_occ, args.occ_emp)

        if kref is None:
            kref = k_m
            klabels = [kpt_label_2x2x2(k) for k in kref]

        assert_same_kmesh(kref, k_m)
        assert_same_kmesh(kref, k_p)

        rows.append(gap_m * Ha_to_eV)
        row_labels.append(f"{conc}  −P")

        rows.append(gap_p * Ha_to_eV)
        row_labels.append(f"{conc}  +P")

    data = np.array(rows)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))

    im = ax.imshow(
        data,
        aspect="auto",
        cmap=args.cmap,
        interpolation="nearest"
    )

    ax.set_xticks(np.arange(len(klabels)))
    ax.set_xticklabels(klabels, fontsize=12)

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=12)

    ax.set_xlabel("k-point", fontsize=13)
    ax.set_ylabel("Concentration / polarization branch", fontsize=13)
    ax.set_title(f"Direct gap heatmap — {args.label}", fontsize=15, pad=12)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Direct gap (eV)", fontsize=12)

    # Cell borders
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Optional numbers inside cells
    if args.annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(
                    j, i, f"{data[i, j]:.2f}",
                    ha="center", va="center",
                    fontsize=8,
                    color="white" if data[i, j] < np.nanmean(data) else "black"
                )
    plt.tight_layout()

    # =========================================================
    # create output folder
    # =========================================================
    outdir = Path("4Direct_Gap")
    outdir.mkdir(exist_ok=True)

    outfile = outdir / Path(args.out).name

    fig.savefig(outfile, dpi=300, bbox_inches="tight")

    print(f"[ok] saved: {outfile}")


if __name__ == "__main__":
    main()
