#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
"""

python3 4make_Zr_all_wfk_HeatMapt.py \
  --dataset "3.125%:/home/yanan/Material/Y_dope_project/3.125Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_MinusP_3.125Zr_HfO2_12247642/berry_MinusP_3o_WFK.nc:/home/yanan/Material/Y_dope_project/3.125Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_PlusP_3.125Zr_HfO2_12247641/berry_PlusP_3o_WFK.nc" \
  --dataset "6.25%:/home/yanan/Material/Y_dope_project/6.25Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_MinusP_6.25Zr_HfO2_12248602/berry_MinusP_6o_WFK.nc:/home/yanan/Material/Y_dope_project/6.25Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_PlusP_6.25Zr_HfO2_12248413/berry_PlusP_6o_WFK.nc" \
  --dataset "9.375%:/home/yanan/Material/Y_dope_project/9.375Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_MinusP_9.375Zr_HfO2_12247637/berry_MinusP_9o_WFK.nc:/home/yanan/Material/Y_dope_project/9.375Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_PlusP_9.375Zr_HfO2_12247629/berry_PlusP_9o_WFK.nc" \
  --out 4Zr_direct_gap_heatmap.png
  
"""
Ha_to_eV = 27.211386245988


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n{' '.join(cmd)}\n\nSTDERR:\n{p.stderr}"
        )
    return p.stdout


def ensure_small_nc(wfk_path: Path, small_path: Path):
    if small_path.exists() and small_path.stat().st_size > 0:
        return

    vars_keep = "reduced_coordinates_of_kpoints,eigenvalues,occupations"

    run([
        "ncks",
        "-O",
        "-v",
        vars_keep,
        str(wfk_path),
        str(small_path)
    ])


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


def compute_gapk_from_small(small_nc, occ_occ=1.5, occ_emp=0.5):

    with Dataset(str(small_nc), "r") as f:

        kpts = np.array(
            f.variables["reduced_coordinates_of_kpoints"][:]
        )

        e = np.array(
            f.variables["eigenvalues"][0, :, :]
        )

        occ = np.array(
            f.variables["occupations"][0, :, :]
        )

    gapk = np.full(e.shape[0], np.nan)

    for ik in range(e.shape[0]):

        occ_idx = np.where(occ[ik] > occ_occ)[0]
        emp_idx = np.where(occ[ik] < occ_emp)[0]

        if len(occ_idx) == 0 or len(emp_idx) == 0:
            continue

        homo = np.max(e[ik, occ_idx])
        lumo = np.min(e[ik, emp_idx])

        gapk[ik] = lumo - homo

    return kpts, gapk * Ha_to_eV


def load_gapk_from_wfk(wfk_path, occ_occ=1.5, occ_emp=0.5):

    wfk_path = wfk_path.expanduser().resolve()

    if not wfk_path.exists():
        raise FileNotFoundError(
            f"Cannot find WFK:\n{wfk_path}"
        )

    small_nc = wfk_path.with_name(
        wfk_path.stem + "_small_bandinfo.nc"
    )

    ensure_small_nc(wfk_path, small_nc)

    return compute_gapk_from_small(
        small_nc,
        occ_occ,
        occ_emp
    )


def assert_same_kmesh(kref, ktest, tol=1e-8):

    if kref.shape != ktest.shape:
        raise ValueError("Different number of k-points.")

    if not np.allclose(kref, ktest, atol=tol, rtol=0):
        raise ValueError("k-point order mismatch.")


def parse_dataset(ds):

    parts = ds.split(":")

    if len(parts) != 3:
        raise ValueError(
            f"Bad dataset format:\n{ds}"
        )

    return (
        parts[0].strip(),
        Path(parts[1].strip()),
        Path(parts[2].strip())
    )


def build_heatmap(dataset_list, occ_occ=1.5, occ_emp=0.5):

    rows = []
    row_labels = []

    kref = None
    klabels = None

    for ds in dataset_list:

        conc, minus_path, plus_path = parse_dataset(ds)

        k_m, gap_m = load_gapk_from_wfk(
            minus_path,
            occ_occ,
            occ_emp
        )

        k_p, gap_p = load_gapk_from_wfk(
            plus_path,
            occ_occ,
            occ_emp
        )

        if kref is None:
            kref = k_m
            klabels = [
                kpt_label_2x2x2(k)
                for k in kref
            ]

        assert_same_kmesh(kref, k_m)
        assert_same_kmesh(kref, k_p)

        rows.append(gap_m)
        row_labels.append(f"{conc}  −P")

        rows.append(gap_p)
        row_labels.append(f"{conc}  +P")

    return np.array(rows), row_labels, klabels


def plot_heatmap(
    data,
    row_labels,
    klabels,
    out,
    title,
    cmap="viridis",
    dpi=600
):

    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    im = ax.imshow(
        data,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(title, fontsize=18, pad=14)

    ax.set_xlabel("k-point", fontsize=15)
    ax.set_ylabel(
        "Concentration / polarization branch",
        fontsize=15
    )

    ax.set_xticks(np.arange(len(klabels)))
    ax.set_xticklabels(klabels, fontsize=14)

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=14)

    ax.set_xticks(
        np.arange(data.shape[1] + 1) - 0.5,
        minor=True
    )

    ax.set_yticks(
        np.arange(data.shape[0] + 1) - 0.5,
        minor=True
    )

    ax.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=1.2
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False
    )

    threshold = (vmin + vmax) / 2.0

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):

            val = data[i, j]

            color = (
                "white"
                if val < threshold
                else "black"
            )

            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=11,
                color=color
            )

    cbar = fig.colorbar(im, ax=ax)

    cbar.set_label(
        "Direct gap (eV)",
        fontsize=15
    )

    cbar.ax.tick_params(labelsize=13)

    plt.tight_layout()

    # ==========================================
    # create output folder
    # ==========================================
    outdir = Path("4Direct_Gap")
    outdir.mkdir(exist_ok=True)

    outfile = outdir / Path(out).name

    fig.savefig(
        outfile,
        dpi=dpi,
        bbox_inches="tight"
    )

    print(f"[ok] saved: {outfile}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        action="append",
        required=True
    )

    parser.add_argument(
        "--out",
        default="Zr_direct_gap_heatmap.png"
    )

    parser.add_argument(
        "--occ_occ",
        type=float,
        default=1.5
    )

    parser.add_argument(
        "--occ_emp",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--cmap",
        default="viridis"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600
    )

    args = parser.parse_args()

    data, row_labels, klabels = build_heatmap(
        args.dataset,
        occ_occ=args.occ_occ,
        occ_emp=args.occ_emp
    )

    plot_heatmap(
        data,
        row_labels,
        klabels,
        out=args.out,
        title="Direct gap heatmap — Zr–HfO$_2$",
        cmap=args.cmap,
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()
