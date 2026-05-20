#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

HARTREE_TO_EV = 27.211386245988

"""
python3 5Bands_WFK_All_Sc_Y_Zr.py \
  --batch-all \
  --show-gap-line \
  --save-npy
"""

def find_var(ds: Dataset, candidates: list[str]) -> str:
    for name in candidates:
        if name in ds.variables:
            return name
    raise KeyError(
        f"Could not find any of {candidates}. "
        f"Available variables: {list(ds.variables.keys())}"
    )


def load_eigenvalues_from_wfk(wfk_path: Path) -> np.ndarray:
    with Dataset(wfk_path, "r") as ds:
        eig_name = find_var(ds, ["eigenvalues", "eig", "Eigenvalues"])
        eig = np.array(ds.variables[eig_name][:]).squeeze()

    if eig.ndim == 2:
        bands = eig
    elif eig.ndim == 3:
        if eig.shape[0] <= 2:
            bands = eig
        elif eig.shape[-1] <= 2:
            bands = np.transpose(eig, (2, 0, 1))
        else:
            raise ValueError(f"Cannot infer eigenvalue shape from {eig.shape}")
    else:
        raise ValueError(f"Unexpected eigenvalue shape: {eig.shape}")

    return bands * HARTREE_TO_EV


def load_occupations_from_wfk(wfk_path: Path) -> np.ndarray | None:
    with Dataset(wfk_path, "r") as ds:
        occ_candidates = ["occupations", "occ", "Occupations"]
        occ_name = None
        for name in occ_candidates:
            if name in ds.variables:
                occ_name = name
                break
        if occ_name is None:
            return None
        occ = np.array(ds.variables[occ_name][:]).squeeze()

    if occ.ndim == 2:
        return occ
    elif occ.ndim == 3:
        if occ.shape[0] <= 2:
            return occ[0]
        elif occ.shape[-1] <= 2:
            return occ[..., 0]
    return None


def find_vbm_cbm_from_occupations(
    bands_ev: np.ndarray,
    occ: np.ndarray,
    occ_threshold: float = 0.5,
):
    if bands_ev.shape != occ.shape:
        raise ValueError(f"bands and occ shape mismatch: {bands_ev.shape} vs {occ.shape}")

    occ_mask = occ > occ_threshold
    emp_mask = occ <= occ_threshold

    if not np.any(occ_mask):
        raise RuntimeError("No occupied states found.")
    if not np.any(emp_mask):
        raise RuntimeError("No empty states found.")

    occ_vals = np.where(occ_mask, bands_ev, -np.inf)
    emp_vals = np.where(emp_mask, bands_ev, +np.inf)

    vbm_flat = int(np.argmax(occ_vals))
    cbm_flat = int(np.argmin(emp_vals))

    vbm_k, vbm_b = np.unravel_index(vbm_flat, bands_ev.shape)
    cbm_k, cbm_b = np.unravel_index(cbm_flat, bands_ev.shape)

    vbm = bands_ev[vbm_k, vbm_b]
    cbm = bands_ev[cbm_k, cbm_b]

    return {
        "vbm_energy_ev": float(vbm),
        "vbm_kpt_index": int(vbm_k + 1),
        "vbm_band_index": int(vbm_b + 1),
        "cbm_energy_ev": float(cbm),
        "cbm_kpt_index": int(cbm_k + 1),
        "cbm_band_index": int(cbm_b + 1),
        "gap_ev": float(cbm - vbm),
    }


def prepare_bands_and_edges(wfk_path: Path, spin: int):
    bands_all = load_eigenvalues_from_wfk(wfk_path)

    if bands_all.ndim == 2:
        bands_ev = bands_all
        nsppol = 1
    else:
        nsppol = bands_all.shape[0]
        if not (1 <= spin <= nsppol):
            raise ValueError(f"Requested spin={spin}, but nsppol={nsppol}")
        bands_ev = bands_all[spin - 1]

    occ = load_occupations_from_wfk(wfk_path)
    edge_info = None
    if occ is not None:
        edge_info = find_vbm_cbm_from_occupations(bands_ev, occ, occ_threshold=0.5)

    return bands_ev, edge_info, nsppol


def get_shift_energy(edge_info, shift_mode: str, manual_vbm: float | None):
    if manual_vbm is not None:
        return manual_vbm
    if shift_mode == "vbm" and edge_info is not None:
        return edge_info["vbm_energy_ev"]
    return 0.0


def draw_one_panel(
    ax,
    bands_ev: np.ndarray,
    title: str,
    energy_shift_ev: float,
    y_min: float | None,
    y_max: float | None,
    k_labels: list[str] | None,
    show_gap_line: bool,
    gap_line_ev: float | None,
    panel_label: str | None = None,
):
    nkpt, nband = bands_ev.shape
    x = np.arange(nkpt)
    y = bands_ev - energy_shift_ev

    for ib in range(nband):
        ax.plot(x, y[:, ib], linewidth=0.8)

    ax.axhline(0.0, linestyle="--", linewidth=1.0)

    if show_gap_line and gap_line_ev is not None:
        ax.axhline(gap_line_ev - energy_shift_ev, linewidth=1.5)

    for xpos in range(nkpt):
        ax.axvline(xpos, linewidth=0.8, alpha=0.35)

    ax.set_xlabel("k-vector", fontsize=13)
    ax.set_ylabel("E (eV)", fontsize=13)
    ax.set_title(title, fontsize=16)

    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)

    ax.set_xticks(range(nkpt))
    if k_labels is not None:
        if len(k_labels) != nkpt:
            raise ValueError(f"Number of k-labels ({len(k_labels)}) must equal nkpt ({nkpt})")
        ax.set_xticklabels(k_labels, fontsize=11)
    else:
        ax.set_xticklabels([str(i + 1) for i in range(nkpt)], fontsize=11)

    ax.tick_params(axis="y", labelsize=11)

    if panel_label is not None:
        ax.text(
            0.02, 0.98, panel_label,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=16, fontweight="bold"
        )


def plot_two_panel_band_structure(
    minus_bands_ev: np.ndarray,
    plus_bands_ev: np.ndarray,
    output_png: Path,
    minus_title: str,
    plus_title: str,
    minus_shift_ev: float,
    plus_shift_ev: float,
    y_min: float | None,
    y_max: float | None,
    k_labels: list[str] | None,
    show_gap_line: bool,
    minus_gap_line_ev: float | None,
    plus_gap_line_ev: float | None,
):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)

    draw_one_panel(
        axes[0], minus_bands_ev, minus_title, minus_shift_ev,
        y_min, y_max, k_labels, show_gap_line, minus_gap_line_ev, panel_label="a)"
    )
    draw_one_panel(
        axes[1], plus_bands_ev, plus_title, plus_shift_ev,
        y_min, y_max, k_labels, show_gap_line, plus_gap_line_ev, panel_label="b)"
    )

    plt.tight_layout()
    plt.savefig(output_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


def print_edge_summary(label: str, edge_info: dict | None):
    print("-" * 80)
    print(label)
    if edge_info is None:
        print("No occupations found; VBM/CBM summary unavailable.")
        return
    print(f"VBM  : {edge_info['vbm_energy_ev']:.6f} eV   at k={edge_info['vbm_kpt_index']}, band={edge_info['vbm_band_index']}")
    print(f"CBM  : {edge_info['cbm_energy_ev']:.6f} eV   at k={edge_info['cbm_kpt_index']}, band={edge_info['cbm_band_index']}")
    print(f"Gap  : {edge_info['gap_ev']:.6f} eV")




# ============================================================
# Batch systems: Sc, Y, Zr at 3.125%, 6.25%, 9.375%
# Each case is saved separately in 5Bands_WFK_All_Results/<tag>/
# ============================================================
SYSTEMS = [
    {
        "tag": "3p125Sc",
        "title": "3.125% Sc-HfO2",
        "minus_wfk": "/home/yanan/Material/Y_dope_project/3.125Per-Sc-HfO2-2x2x2/berry-scf-Results/berry_MinusP_3.125Sc_HfO2_12144332/berry_MinusP_3o_WFK.nc",
        "plus_wfk":  "/home/yanan/Material/Y_dope_project/3.125Per-Sc-HfO2-2x2x2/berry-scf-Results/berry_PlusP_3.125Sc_HfO2_12144296/berry_PlusP_3o_WFK.nc",
    },
    {
        "tag": "6p25Sc",
        "title": "6.25% Sc-HfO2",
        "minus_wfk": "/home/yanan/Material/Y_dope_project/6.25Per-Sc-HfO2-2x2x2/berry-scf-results/berry_MinusP_6.25Sc_HfO2_12144457/berry_MinusP_6o_WFK.nc",
        "plus_wfk":  "/home/yanan/Material/Y_dope_project/6.25Per-Sc-HfO2-2x2x2/berry-scf-results/berry_PlusP_6.25Sc_HfO2_12144349/berry_PlusP_6o_WFK.nc",
    },
    {
        "tag": "9p375Sc",
        "title": "9.375% Sc-HfO2",
        "minus_wfk": "/home/yanan/Material/Y_dope_project/9.375Sc-HfO2-2x2x2/berry-scf-Results/berry_MinusP_9.375Sc_HfO2_12144529/berry_MinusP_9o_WFK.nc",
        "plus_wfk":  "/home/yanan/Material/Y_dope_project/9.375Sc-HfO2-2x2x2/berry-scf-Results/berry_PlusP_9.375Sc_HfO2_12144507/berry_PlusP_9o_WFK.nc",
    },
    {
        "tag": "3p125Y",
        "title": "3.125% Y-HfO2",
        "minus_wfk": "/home/yanan/Material/Y_dope_project/3.125Per-Y-HfO2-2x2x2/berry-scf-results/berry_MinusP_3.125Per_Y_HfO2_12118864/berry_MinusP_3o_WFK.nc",
        "plus_wfk":  "/home/yanan/Material/Y_dope_project/3.125Per-Y-HfO2-2x2x2/berry-scf-results/berry_PlusP_3.125Per_Y_HfO2_12118862/berry_PlusP_3o_WFK.nc",
    },
    {
        "tag": "6p25Y",
        "title": "6.25% Y-HfO2",
        "minus_wfk": "/home/yanan/Material/Y_dope_project/6.25Per-Y-HfO2-2x2x2/berry-scf-results/berry_MinusP_6.25Y_HfO2_12397618/berry_MinusP_6o_WFK.nc",
        "plus_wfk":  "/home/yanan/Material/Y_dope_project/6.25Per-Y-HfO2-2x2x2/berry-scf-results/berry_PlusP_6.25Y_HfO2_12651723/berry_PlusP_6o_WFK.nc",
    },
    {
        "tag": "9p375Y",
        "title": "9.375% Y-HfO2",
        "minus_wfk": "/home/yanan/Material/Y_dope_project/9.375Y-HfO2-2x2x2/9.375Y-scf-berry-2x2x2-Results/berry_MinusP_9.375Y_HfO2_12121996/berry_MinusP_9o_WFK.nc",
        "plus_wfk":  "/home/yanan/Material/Y_dope_project/9.375Y-HfO2-2x2x2/9.375Y-scf-berry-2x2x2-Results/berry_PlusP_9.375Y_HfO2_12122002/berry_PlusP_9o_WFK.nc",
    },
    {
        "tag": "3p125Zr",
        "title": "3.125% Zr-HfO2",
        "minus_wfk": "/home/yanan/Material/Y_dope_project/3.125Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_MinusP_3.125Zr_HfO2_12247642/berry_MinusP_3o_WFK.nc",
        "plus_wfk":  "/home/yanan/Material/Y_dope_project/3.125Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_PlusP_3.125Zr_HfO2_12247641/berry_PlusP_3o_WFK.nc",
    },
    {
        "tag": "6p25Zr",
        "title": "6.25% Zr-HfO2",
        "minus_wfk": "/home/yanan/Material/Y_dope_project/6.25Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_MinusP_6.25Zr_HfO2_12248602/berry_MinusP_6o_WFK.nc",
        "plus_wfk":  "/home/yanan/Material/Y_dope_project/6.25Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_PlusP_6.25Zr_HfO2_12248413/berry_PlusP_6o_WFK.nc",
    },
    {
        "tag": "9p375Zr",
        "title": "9.375% Zr-HfO2",
        "minus_wfk": "/home/yanan/Material/Y_dope_project/9.375Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_MinusP_9.375Zr_HfO2_12247637/berry_MinusP_9o_WFK.nc",
        "plus_wfk":  "/home/yanan/Material/Y_dope_project/9.375Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_PlusP_9.375Zr_HfO2_12247629/berry_PlusP_9o_WFK.nc",
    },
]


def run_one_case(
    minus_wfk: Path,
    plus_wfk: Path,
    minus_title: str,
    plus_title: str,
    output: Path,
    save_minus_npy: Path | None = None,
    save_plus_npy: Path | None = None,
    spin: int = 1,
    shift_mode: str = "vbm",
    minus_vbm_ev: float | None = None,
    plus_vbm_ev: float | None = None,
    ymin: float | None = -2.0,
    ymax: float | None = 8.0,
    show_gap_line: bool = True,
    klabels: str | None = "Γ,X,Y,S,Z,U,T,R",
):
    output.parent.mkdir(parents=True, exist_ok=True)
    if save_minus_npy is not None:
        save_minus_npy.parent.mkdir(parents=True, exist_ok=True)
    if save_plus_npy is not None:
        save_plus_npy.parent.mkdir(parents=True, exist_ok=True)

    minus_bands_ev, minus_edge, minus_nsppol = prepare_bands_and_edges(minus_wfk, spin)
    plus_bands_ev, plus_edge, plus_nsppol = prepare_bands_and_edges(plus_wfk, spin)

    if minus_bands_ev.shape != plus_bands_ev.shape:
        raise ValueError(f"MinusP and PlusP bands shape mismatch: {minus_bands_ev.shape} vs {plus_bands_ev.shape}")

    minus_shift = get_shift_energy(minus_edge, shift_mode, minus_vbm_ev)
    plus_shift = get_shift_energy(plus_edge, shift_mode, plus_vbm_ev)

    if save_minus_npy is not None:
        np.save(save_minus_npy, minus_bands_ev)
    if save_plus_npy is not None:
        np.save(save_plus_npy, plus_bands_ev)

    k_labels = [s.strip() for s in klabels.split(",")] if klabels else None

    minus_gap_line = minus_edge["cbm_energy_ev"] if (show_gap_line and minus_edge is not None) else None
    plus_gap_line = plus_edge["cbm_energy_ev"] if (show_gap_line and plus_edge is not None) else None

    plot_two_panel_band_structure(
        minus_bands_ev=minus_bands_ev,
        plus_bands_ev=plus_bands_ev,
        output_png=output,
        minus_title=minus_title,
        plus_title=plus_title,
        minus_shift_ev=minus_shift,
        plus_shift_ev=plus_shift,
        y_min=ymin,
        y_max=ymax,
        k_labels=k_labels,
        show_gap_line=show_gap_line,
        minus_gap_line_ev=minus_gap_line,
        plus_gap_line_ev=plus_gap_line,
    )

    summary_path = output.parent / "band_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"MinusP file : {minus_wfk}\n")
        f.write(f"PlusP file  : {plus_wfk}\n")
        f.write(f"Minus nsppol: {minus_nsppol}\n")
        f.write(f"Plus nsppol : {plus_nsppol}\n")
        f.write(f"Shape       : {minus_bands_ev.shape}\n")
        f.write(f"Shift mode  : {shift_mode}\n")
        f.write(f"Minus shift : {minus_shift:.6f} eV\n")
        f.write(f"Plus shift  : {plus_shift:.6f} eV\n\n")
        for label, edge in [("MinusP", minus_edge), ("PlusP", plus_edge)]:
            f.write(f"{label}\n")
            if edge is None:
                f.write("No occupations found; VBM/CBM summary unavailable.\n")
            else:
                f.write(f"VBM  : {edge['vbm_energy_ev']:.6f} eV at k={edge['vbm_kpt_index']}, band={edge['vbm_band_index']}\n")
                f.write(f"CBM  : {edge['cbm_energy_ev']:.6f} eV at k={edge['cbm_kpt_index']}, band={edge['cbm_band_index']}\n")
                f.write(f"Gap  : {edge['gap_ev']:.6f} eV\n")
            f.write("\n")

    print("=" * 80)
    print(f"Output      : {output}")
    print(f"Summary     : {summary_path}")
    print_edge_summary("MinusP", minus_edge)
    print_edge_summary("PlusP", plus_edge)
    print("=" * 80)


def run_batch_all(args):
    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)

    completed = 0
    failures = []

    # One global summary text file for all systems
    global_summary = root / "all_band_gaps.txt"
    with open(global_summary, "w") as f:
        f.write("System  State   VBM(eV)   VBM_k/b   CBM(eV)   CBM_k/b   Gap(eV)\n")
        f.write("=" * 88 + "\n")

    for item in SYSTEMS:
        tag = item["tag"]
        title = item["title"]
        case_dir = root / tag
        output = case_dir / f"band_{tag}_minus_plus.png"
        save_minus = case_dir / f"bands_{tag}_MinusP.npy" if args.save_npy else None
        save_plus = case_dir / f"bands_{tag}_PlusP.npy" if args.save_npy else None

        minus_wfk = Path(item["minus_wfk"])
        plus_wfk = Path(item["plus_wfk"])

        if not minus_wfk.exists():
            reason = f"Missing MinusP WFK: {minus_wfk}"
            failures.append((tag, reason))
            print(f"Missing MinusP WFK for {tag}: {minus_wfk}")
            continue

        if not plus_wfk.exists():
            reason = f"Missing PlusP WFK: {plus_wfk}"
            failures.append((tag, reason))
            print(f"Missing PlusP WFK for {tag}: {plus_wfk}")
            continue

        try:
            # Run band plotting and write the per-system band_summary.txt
            run_one_case(
                minus_wfk=minus_wfk,
                plus_wfk=plus_wfk,
                minus_title=f"{title} MinusP",
                plus_title=f"{title} PlusP",
                output=output,
                save_minus_npy=save_minus,
                save_plus_npy=save_plus,
                spin=args.spin,
                shift_mode=args.shift_mode,
                ymin=args.ymin,
                ymax=args.ymax,
                show_gap_line=args.show_gap_line,
                klabels=args.klabels,
            )

            # Re-read edge information and append it to one global text file.
            # This gives a single table for Sc/Y/Zr at 3.125%, 6.25%, and 9.375%.
            _, minus_edge, _ = prepare_bands_and_edges(minus_wfk, args.spin)
            _, plus_edge, _ = prepare_bands_and_edges(plus_wfk, args.spin)

            with open(global_summary, "a") as f:
                if minus_edge is not None:
                    f.write(
                        f"{tag:<9s} MinusP  "
                        f"{minus_edge['vbm_energy_ev']:>10.6f}  "
                        f"({minus_edge['vbm_kpt_index']},{minus_edge['vbm_band_index']})  "
                        f"{minus_edge['cbm_energy_ev']:>10.6f}  "
                        f"({minus_edge['cbm_kpt_index']},{minus_edge['cbm_band_index']})  "
                        f"{minus_edge['gap_ev']:>10.6f}\n"
                    )
                else:
                    f.write(f"{tag:<9s} MinusP  No occupation data available\n")

                if plus_edge is not None:
                    f.write(
                        f"{tag:<9s} PlusP   "
                        f"{plus_edge['vbm_energy_ev']:>10.6f}  "
                        f"({plus_edge['vbm_kpt_index']},{plus_edge['vbm_band_index']})  "
                        f"{plus_edge['cbm_energy_ev']:>10.6f}  "
                        f"({plus_edge['cbm_kpt_index']},{plus_edge['cbm_band_index']})  "
                        f"{plus_edge['gap_ev']:>10.6f}\n"
                    )
                else:
                    f.write(f"{tag:<9s} PlusP   No occupation data available\n")

                f.write("\n")

            completed += 1

        except Exception as exc:
            failures.append((tag, str(exc)))
            print(f"FAILED {tag}: {exc}")

    batch_summary = root / "batch_summary.txt"
    with open(batch_summary, "w") as f:
        f.write(f"Completed: {completed}/{len(SYSTEMS)}\n")
        f.write(f"Global band-gap text file: {global_summary}\n")
        if failures:
            f.write("\nFailures:\n")
            for tag, reason in failures:
                f.write(f"{tag}: {reason}\n")

    print("\n" + "=" * 80)
    print(f"Completed: {completed}/{len(SYSTEMS)}")
    print(f"Batch summary: {batch_summary}")
    print(f"Global text file: {global_summary}")
    if failures:
        print("Failures:")
        for tag, reason in failures:
            print(f"  {tag}: {reason}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Plot MinusP and PlusP band structures from ABINIT WFK NetCDF")

    parser.add_argument("--batch-all", action="store_true", help="Run Sc/Y/Zr at 3.125%, 6.25%, 9.375% separately")
    parser.add_argument("--outdir", type=Path, default=Path("5Bands_WFK_All_Results"), help="Output root directory")

    # single-pair mode
    parser.add_argument("--minus-wfk", type=Path, help="MinusP WFK path")
    parser.add_argument("--plus-wfk", type=Path, help="PlusP WFK path")
    parser.add_argument("--minus-title", type=str, default="MinusP", help="Left panel title")
    parser.add_argument("--plus-title", type=str, default="PlusP", help="Right panel title")
    parser.add_argument("--output", type=Path, default=Path("band_comparison_minus_plus.png"), help="Output PNG for single mode")
    parser.add_argument("--save-minus-npy", type=Path, default=None, help="Optional MinusP .npy output in single mode")
    parser.add_argument("--save-plus-npy", type=Path, default=None, help="Optional PlusP .npy output in single mode")

    # common options
    parser.add_argument("--save-npy", action="store_true", help="In batch mode, save separate .npy arrays for each system")
    parser.add_argument("--spin", type=int, default=1, help="Spin channel to plot (1-based)")
    parser.add_argument("--shift-mode", choices=["none", "vbm"], default="vbm", help="Energy shift mode")
    parser.add_argument("--ymin", type=float, default=-2.0, help="Lower y-limit after shifting")
    parser.add_argument("--ymax", type=float, default=8.0, help="Upper y-limit after shifting")
    parser.add_argument("--show-gap-line", action="store_true", help="Draw horizontal CBM line in each panel")
    parser.add_argument("--klabels", type=str, default="Γ,X,Y,S,Z,U,T,R", help='Comma-separated labels, e.g. "Γ,X,Y,S,Z,U,T,R"')

    args = parser.parse_args()

    if args.batch_all:
        run_batch_all(args)
        return

    if args.minus_wfk is None or args.plus_wfk is None:
        raise SystemExit("Single mode requires --minus-wfk and --plus-wfk, or use --batch-all.")

    out = args.outdir / args.output.name
    minus_npy = args.outdir / args.save_minus_npy.name if args.save_minus_npy else None
    plus_npy = args.outdir / args.save_plus_npy.name if args.save_plus_npy else None

    run_one_case(
        minus_wfk=args.minus_wfk,
        plus_wfk=args.plus_wfk,
        minus_title=args.minus_title,
        plus_title=args.plus_title,
        output=out,
        save_minus_npy=minus_npy,
        save_plus_npy=plus_npy,
        spin=args.spin,
        shift_mode=args.shift_mode,
        ymin=args.ymin,
        ymax=args.ymax,
        show_gap_line=args.show_gap_line,
        klabels=args.klabels,
    )


if __name__ == "__main__":
    main()
