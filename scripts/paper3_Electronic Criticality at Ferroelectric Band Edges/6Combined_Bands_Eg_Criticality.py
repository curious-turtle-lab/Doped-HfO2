#!/usr/bin/env python3
"""
Create one combined 3x3 band-structure figure for Sc-, Y-, and Zr-doped HfO2
at 3.125%, 6.25%, and 9.375%, and extract gap metrics:

    Eg(+P), Eg(-P), Delta Eg = Eg(+P) - Eg(-P), and A_Eg.

Rows    : Sc, Y, Zr
Columns : 3.125%, 6.25%, 9.375%
Each panel overlays MinusP and PlusP band structures on the same VBM-shifted axis.

Run:
    python3 6Combined_Bands_Eg_Criticality.py \
        --outdir 6combined_band_criticality_results \
        --ymin -0.6 --ymax 5.0

Outputs:
    6combined_band_criticality_results/combined_3x3_bands.png
    6combined_band_criticality_results/gap_metrics.csv
    6combined_band_criticality_results/gap_metrics.txt
    6combined_band_criticality_results/criticality_summary.txt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

HARTREE_TO_EV = 27.211386245988

# ============================================================
# Systems: Sc, Y, Zr at 3.125%, 6.25%, 9.375%
# ============================================================
SYSTEMS: List[Dict[str, str | float]] = [
    # ---------------- Sc ----------------
    {
        "dopant": "Sc",
        "doping": 3.125,
        "tag": "3p125Sc",
        "title": "3.125% Sc-HfO$_2$",
        "minus_wfk": "/home/yanan/Material/Y_dope_project/3.125Per-Sc-HfO2-2x2x2/berry-scf-Results/berry_MinusP_3.125Sc_HfO2_12144332/berry_MinusP_3o_WFK.nc",
        "plus_wfk": "/home/yanan/Material/Y_dope_project/3.125Per-Sc-HfO2-2x2x2/berry-scf-Results/berry_PlusP_3.125Sc_HfO2_12144296/berry_PlusP_3o_WFK.nc",
    },
    {
        "dopant": "Sc",
        "doping": 6.25,
        "tag": "6p25Sc",
        "title": "6.25% Sc-HfO$_2$",
        "minus_wfk": "/home/yanan/Material/Y_dope_project/6.25Per-Sc-HfO2-2x2x2/berry-scf-results/berry_MinusP_6.25Sc_HfO2_12144457/berry_MinusP_6o_WFK.nc",
        "plus_wfk": "/home/yanan/Material/Y_dope_project/6.25Per-Sc-HfO2-2x2x2/berry-scf-results/berry_PlusP_6.25Sc_HfO2_12144349/berry_PlusP_6o_WFK.nc",
    },
    {
        "dopant": "Sc",
        "doping": 9.375,
        "tag": "9p375Sc",
        "title": "9.375% Sc-HfO$_2$",
        "minus_wfk": "/home/yanan/Material/Y_dope_project/9.375Sc-HfO2-2x2x2/berry-scf-Results/berry_MinusP_9.375Sc_HfO2_12144529/berry_MinusP_9o_WFK.nc",
        "plus_wfk": "/home/yanan/Material/Y_dope_project/9.375Sc-HfO2-2x2x2/berry-scf-Results/berry_PlusP_9.375Sc_HfO2_12144507/berry_PlusP_9o_WFK.nc",
    },

    # ---------------- Y ----------------
    {
        "dopant": "Y",
        "doping": 3.125,
        "tag": "3p125Y",
        "title": "3.125% Y-HfO$_2$",
        "minus_wfk": "/home/yanan/Material/Y_dope_project/3.125Per-Y-HfO2-2x2x2/berry-scf-results/berry_MinusP_3.125Per_Y_HfO2_12118864/berry_MinusP_3o_WFK.nc",
        "plus_wfk": "/home/yanan/Material/Y_dope_project/3.125Per-Y-HfO2-2x2x2/berry-scf-results/berry_PlusP_3.125Per_Y_HfO2_12118862/berry_PlusP_3o_WFK.nc",
    },
    {
        "dopant": "Y",
        "doping": 6.25,
        "tag": "6p25Y",
        "title": "6.25% Y-HfO$_2$",
        "minus_wfk": "/home/yanan/Material/Y_dope_project/6.25Per-Y-HfO2-2x2x2/berry-scf-results/berry_MinusP_6.25Y_HfO2_12397618/berry_MinusP_6o_WFK.nc",
        "plus_wfk": "/home/yanan/Material/Y_dope_project/6.25Per-Y-HfO2-2x2x2/berry-scf-results/berry_PlusP_6.25Y_HfO2_12651723/berry_PlusP_6o_WFK.nc",
    },
    {
        "dopant": "Y",
        "doping": 9.375,
        "tag": "9p375Y",
        "title": "9.375% Y-HfO$_2$",
        "minus_wfk": "/home/yanan/Material/Y_dope_project/9.375Y-HfO2-2x2x2/9.375Y-scf-berry-2x2x2-Results/berry_MinusP_9.375Y_HfO2_12121996/berry_MinusP_9o_WFK.nc",
        "plus_wfk": "/home/yanan/Material/Y_dope_project/9.375Y-HfO2-2x2x2/9.375Y-scf-berry-2x2x2-Results/berry_PlusP_9.375Y_HfO2_12122002/berry_PlusP_9o_WFK.nc",
    },

    # ---------------- Zr ----------------
    {
        "dopant": "Zr",
        "doping": 3.125,
        "tag": "3p125Zr",
        "title": "3.125% Zr-HfO$_2$",
        "minus_wfk": "/home/yanan/Material/Y_dope_project/3.125Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_MinusP_3.125Zr_HfO2_12247642/berry_MinusP_3o_WFK.nc",
        "plus_wfk": "/home/yanan/Material/Y_dope_project/3.125Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_PlusP_3.125Zr_HfO2_12247641/berry_PlusP_3o_WFK.nc",
    },
    {
        "dopant": "Zr",
        "doping": 6.25,
        "tag": "6p25Zr",
        "title": "6.25% Zr-HfO$_2$",
        "minus_wfk": "/home/yanan/Material/Y_dope_project/6.25Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_MinusP_6.25Zr_HfO2_12248602/berry_MinusP_6o_WFK.nc",
        "plus_wfk": "/home/yanan/Material/Y_dope_project/6.25Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_PlusP_6.25Zr_HfO2_12248413/berry_PlusP_6o_WFK.nc",
    },
    {
        "dopant": "Zr",
        "doping": 9.375,
        "tag": "9p375Zr",
        "title": "9.375% Zr-HfO$_2$",
        "minus_wfk": "/home/yanan/Material/Y_dope_project/9.375Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_MinusP_9.375Zr_HfO2_12247637/berry_MinusP_9o_WFK.nc",
        "plus_wfk": "/home/yanan/Material/Y_dope_project/9.375Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_PlusP_9.375Zr_HfO2_12247629/berry_PlusP_9o_WFK.nc",
    },
]

DOPANT_ORDER = ["Sc", "Y", "Zr"]
DOPING_ORDER = [3.125, 6.25, 9.375]


# ============================================================
# WFK reading and band-edge extraction
# ============================================================
def find_var(ds: Dataset, candidates: List[str]) -> str:
    for name in candidates:
        if name in ds.variables:
            return name
    raise KeyError(
        f"Could not find any of {candidates}. Available variables: {list(ds.variables.keys())}"
    )


def load_eigenvalues_from_wfk(wfk_path: Path) -> np.ndarray:
    with Dataset(wfk_path, "r") as ds:
        eig_name = find_var(ds, ["eigenvalues", "eig", "Eigenvalues"])
        eig = np.array(ds.variables[eig_name][:]).squeeze()

    if eig.ndim == 2:
        bands = eig
    elif eig.ndim == 3:
        # Shape may be (nsppol, nkpt, nband) or (nkpt, nband, nsppol)
        if eig.shape[0] <= 2:
            bands = eig
        elif eig.shape[-1] <= 2:
            bands = np.transpose(eig, (2, 0, 1))
        else:
            raise ValueError(f"Cannot infer eigenvalue shape from {eig.shape}")
    else:
        raise ValueError(f"Unexpected eigenvalue shape: {eig.shape}")

    return bands * HARTREE_TO_EV


def load_occupations_from_wfk(wfk_path: Path) -> Optional[np.ndarray]:
    with Dataset(wfk_path, "r") as ds:
        occ_name = None
        for name in ["occupations", "occ", "Occupations"]:
            if name in ds.variables:
                occ_name = name
                break
        if occ_name is None:
            return None
        occ = np.array(ds.variables[occ_name][:]).squeeze()

    if occ.ndim == 2:
        return occ
    if occ.ndim == 3:
        if occ.shape[0] <= 2:
            return occ[0]
        if occ.shape[-1] <= 2:
            return occ[..., 0]
    return None


def get_spin_bands(bands_all: np.ndarray, spin: int) -> Tuple[np.ndarray, int]:
    if bands_all.ndim == 2:
        return bands_all, 1

    nsppol = bands_all.shape[0]
    if not (1 <= spin <= nsppol):
        raise ValueError(f"Requested spin={spin}, but nsppol={nsppol}")
    return bands_all[spin - 1], nsppol


def find_vbm_cbm_from_occupations(
    bands_ev: np.ndarray,
    occ: np.ndarray,
    occ_threshold: float = 0.5,
) -> Dict[str, float | int]:
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

    vbm = float(bands_ev[vbm_k, vbm_b])
    cbm = float(bands_ev[cbm_k, cbm_b])

    return {
        "vbm_energy_ev": vbm,
        "vbm_kpt_index": int(vbm_k + 1),
        "vbm_band_index": int(vbm_b + 1),
        "cbm_energy_ev": cbm,
        "cbm_kpt_index": int(cbm_k + 1),
        "cbm_band_index": int(cbm_b + 1),
        "gap_ev": float(cbm - vbm),
    }


def prepare_bands_and_edges(wfk_path: Path, spin: int) -> Tuple[np.ndarray, Dict[str, float | int], int]:
    bands_all = load_eigenvalues_from_wfk(wfk_path)
    bands_ev, nsppol = get_spin_bands(bands_all, spin)

    occ = load_occupations_from_wfk(wfk_path)
    if occ is None:
        raise RuntimeError(f"No occupations found in {wfk_path}. Cannot identify VBM/CBM.")

    edge_info = find_vbm_cbm_from_occupations(bands_ev, occ, occ_threshold=0.5)
    return bands_ev, edge_info, nsppol


def compute_AEg(eg_plus: float, eg_minus: float) -> float:
    denom = eg_plus + eg_minus
    if abs(denom) < 1e-14:
        return np.nan
    return float((eg_plus - eg_minus) / denom)


def select_window_around_vbm(
    bands_ev: np.ndarray,
    vbm_ev: float,
    window_min: float,
    window_max: float,
) -> np.ndarray:
    """Return only bands that enter the plotting energy window after VBM shifting."""
    shifted = bands_ev - vbm_ev
    keep = np.any((shifted >= window_min) & (shifted <= window_max), axis=0)
    if not np.any(keep):
        return shifted
    return shifted[:, keep]


# ============================================================
# Plotting
# ============================================================
def draw_overlay_panel(
    ax,
    minus_bands_ev: np.ndarray,
    plus_bands_ev: np.ndarray,
    minus_edge: Dict[str, float | int],
    plus_edge: Dict[str, float | int],
    title: str,
    tag: str,
    ymin: float,
    ymax: float,
    klabels: List[str],
):
    nkpt, _ = minus_bands_ev.shape
    x = np.arange(nkpt)

    minus_shift = float(minus_edge["vbm_energy_ev"])
    plus_shift = float(plus_edge["vbm_energy_ev"])

    minus_plot = select_window_around_vbm(minus_bands_ev, minus_shift, ymin, ymax)
    plus_plot = select_window_around_vbm(plus_bands_ev, plus_shift, ymin, ymax)

    # MinusP: solid; PlusP: dashed. Use default matplotlib colors.
    for ib in range(minus_plot.shape[1]):
        ax.plot(x, minus_plot[:, ib], linewidth=0.55, alpha=0.78)
    for ib in range(plus_plot.shape[1]):
        ax.plot(x, plus_plot[:, ib], linewidth=0.55, linestyle="--", alpha=0.78)

    ax.axhline(0.0, linestyle=":", linewidth=1.0)

    # CBM lines after each state's own VBM shift.
    eg_minus = float(minus_edge["gap_ev"])
    eg_plus = float(plus_edge["gap_ev"])
    ax.axhline(eg_minus, linewidth=1.1, linestyle="-")
    ax.axhline(eg_plus, linewidth=1.1, linestyle="--")

    for xpos in range(nkpt):
        ax.axvline(xpos, linewidth=0.45, alpha=0.25)

    delta_eg = eg_plus - eg_minus
    A_eg = compute_AEg(eg_plus, eg_minus)

    ax.set_title(title, fontsize=11)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0, nkpt - 1)
    ax.set_xticks(range(nkpt))
    ax.set_xticklabels(klabels, fontsize=8)
    ax.tick_params(axis="y", labelsize=8)

    text = (
        rf"$E_g^{{-P}}$={eg_minus:.3f} eV\n"
        rf"$E_g^{{+P}}$={eg_plus:.3f} eV\n"
        rf"$\Delta E_g$={delta_eg:.3f} eV\n"
        rf"$A_{{E_g}}$={A_eg:.3f}"
    )
    ax.text(
        0.03, 0.97, text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.80, linewidth=0.4),
    )

    # Small legend proxy only once per panel, simple and readable.
    ax.plot([], [], linewidth=1.0, linestyle="-", label="MinusP")
    ax.plot([], [], linewidth=1.0, linestyle="--", label="PlusP")
    ax.legend(fontsize=7, loc="lower right", frameon=True)


def create_combined_3x3_figure(
    data: Dict[Tuple[str, float], Dict],
    outpath: Path,
    ymin: float,
    ymax: float,
    klabels: List[str],
):
    fig, axes = plt.subplots(3, 3, figsize=(15.0, 11.0), sharex=True, sharey=True)

    for i, dopant in enumerate(DOPANT_ORDER):
        for j, doping in enumerate(DOPING_ORDER):
            ax = axes[i, j]
            item = data[(dopant, doping)]
            draw_overlay_panel(
                ax=ax,
                minus_bands_ev=item["minus_bands_ev"],
                plus_bands_ev=item["plus_bands_ev"],
                minus_edge=item["minus_edge"],
                plus_edge=item["plus_edge"],
                title=item["title"],
                tag=item["tag"],
                ymin=ymin,
                ymax=ymax,
                klabels=klabels,
            )

            if j == 0:
                ax.set_ylabel(f"{dopant}\nE - VBM (eV)", fontsize=11)
            if i == 2:
                ax.set_xlabel("k-path", fontsize=10)

    fig.suptitle(
        "Polarization-stabilized electronic criticality in Sc-, Y-, and Zr-doped HfO$_2$",
        fontsize=16,
        y=0.995,
    )
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.975])
    fig.savefig(outpath, dpi=600, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Output tables and interpretation text
# ============================================================
def write_gap_metrics(rows: List[Dict], outdir: Path):
    csv_path = outdir / "gap_metrics.csv"
    txt_path = outdir / "gap_metrics.txt"

    fieldnames = [
        "dopant", "doping_percent", "tag",
        "Eg_plus_eV", "Eg_minus_eV", "Delta_Eg_eV", "A_Eg",
        "PlusP_VBM_eV", "PlusP_CBM_eV", "PlusP_VBM_k", "PlusP_VBM_band", "PlusP_CBM_k", "PlusP_CBM_band",
        "MinusP_VBM_eV", "MinusP_CBM_eV", "MinusP_VBM_k", "MinusP_VBM_band", "MinusP_CBM_k", "MinusP_CBM_band",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(txt_path, "w") as f:
        f.write("Band-gap asymmetry metrics for polarization-stabilized electronic criticality\n")
        f.write("=" * 86 + "\n\n")
        f.write(
            f"{'Dopant':<7s} {'Doping(%)':>10s} {'Eg(+P)':>12s} {'Eg(-P)':>12s} "
            f"{'DeltaEg':>12s} {'A_Eg':>12s}\n"
        )
        f.write("-" * 86 + "\n")
        for r in rows:
            f.write(
                f"{r['dopant']:<7s} {r['doping_percent']:>10.3f} "
                f"{r['Eg_plus_eV']:>12.6f} {r['Eg_minus_eV']:>12.6f} "
                f"{r['Delta_Eg_eV']:>12.6f} {r['A_Eg']:>12.6f}\n"
            )

    return csv_path, txt_path


def write_criticality_summary(rows: List[Dict], outdir: Path):
    summary_path = outdir / "criticality_summary.txt"

    # Sort by absolute average gap. Smaller |Eg_avg| is more critical / closer to band closure.
    enriched = []
    for r in rows:
        eg_avg = 0.5 * (float(r["Eg_plus_eV"]) + float(r["Eg_minus_eV"]))
        enriched.append((abs(eg_avg), eg_avg, r))
    enriched.sort(key=lambda x: x[0])

    with open(summary_path, "w") as f:
        f.write("Polarization-stabilized electronic criticality summary\n")
        f.write("=" * 62 + "\n\n")
        f.write("Definitions\n")
        f.write("-----------\n")
        f.write("Eg(+P)  : fundamental gap extracted from the PlusP WFK occupations.\n")
        f.write("Eg(-P)  : fundamental gap extracted from the MinusP WFK occupations.\n")
        f.write("DeltaEg : Eg(+P) - Eg(-P).\n")
        f.write("A_Eg    : [Eg(+P) - Eg(-P)] / [Eg(+P) + Eg(-P)].\n\n")

        f.write("Criticality ranking by |average gap|\n")
        f.write("------------------------------------\n")
        for abs_avg, eg_avg, r in enriched:
            f.write(
                f"{r['tag']:<9s}  Eg_avg={eg_avg: .6f} eV  "
                f"|Eg_avg|={abs_avg:.6f} eV  "
                f"DeltaEg={r['Delta_Eg_eV']: .6f} eV  "
                f"A_Eg={r['A_Eg']: .6f}\n"
            )

        f.write("\nInterpretation paragraph for manuscript draft\n")
        f.write("---------------------------------------------\n")
        f.write(
            "The combined band-structure analysis shows how polarization reversal reshapes the near-edge "
            "electronic spectrum across dopant species and concentration. The extracted quantities Eg(+P), "
            "Eg(-P), DeltaEg, and A_Eg quantify the polarization sensitivity of the fundamental gap. "
            "Systems with small |Eg| or negative Eg are close to band closure and therefore represent an "
            "electronically critical regime. In this picture, polarization-stabilized electronic criticality "
            "means that the ferroelectric polarization state does not merely switch ionic displacements; it "
            "also tunes the band-edge alignment and can drive the material toward or away from a small-gap, "
            "high-leakage-risk electronic structure. The dopant-dependent trend therefore links local chemical "
            "substitution, polarization reversal, and electronic-gap asymmetry in a single mechanism.\n"
        )

    return summary_path


def collect_all_data(spin: int) -> Tuple[Dict[Tuple[str, float], Dict], List[Dict]]:
    data: Dict[Tuple[str, float], Dict] = {}
    rows: List[Dict] = []

    for system in SYSTEMS:
        tag = str(system["tag"])
        dopant = str(system["dopant"])
        doping = float(system["doping"])
        title = str(system["title"])
        minus_wfk = Path(str(system["minus_wfk"]))
        plus_wfk = Path(str(system["plus_wfk"]))

        if not minus_wfk.exists():
            raise FileNotFoundError(f"Missing MinusP WFK for {tag}: {minus_wfk}")
        if not plus_wfk.exists():
            raise FileNotFoundError(f"Missing PlusP WFK for {tag}: {plus_wfk}")

        minus_bands_ev, minus_edge, _ = prepare_bands_and_edges(minus_wfk, spin)
        plus_bands_ev, plus_edge, _ = prepare_bands_and_edges(plus_wfk, spin)

        if minus_bands_ev.shape != plus_bands_ev.shape:
            raise ValueError(
                f"Bands shape mismatch for {tag}: {minus_bands_ev.shape} vs {plus_bands_ev.shape}"
            )

        eg_minus = float(minus_edge["gap_ev"])
        eg_plus = float(plus_edge["gap_ev"])
        delta_eg = eg_plus - eg_minus
        A_eg = compute_AEg(eg_plus, eg_minus)

        data[(dopant, doping)] = {
            "tag": tag,
            "dopant": dopant,
            "doping": doping,
            "title": title,
            "minus_bands_ev": minus_bands_ev,
            "plus_bands_ev": plus_bands_ev,
            "minus_edge": minus_edge,
            "plus_edge": plus_edge,
        }

        rows.append({
            "dopant": dopant,
            "doping_percent": doping,
            "tag": tag,
            "Eg_plus_eV": eg_plus,
            "Eg_minus_eV": eg_minus,
            "Delta_Eg_eV": delta_eg,
            "A_Eg": A_eg,
            "PlusP_VBM_eV": float(plus_edge["vbm_energy_ev"]),
            "PlusP_CBM_eV": float(plus_edge["cbm_energy_ev"]),
            "PlusP_VBM_k": int(plus_edge["vbm_kpt_index"]),
            "PlusP_VBM_band": int(plus_edge["vbm_band_index"]),
            "PlusP_CBM_k": int(plus_edge["cbm_kpt_index"]),
            "PlusP_CBM_band": int(plus_edge["cbm_band_index"]),
            "MinusP_VBM_eV": float(minus_edge["vbm_energy_ev"]),
            "MinusP_CBM_eV": float(minus_edge["cbm_energy_ev"]),
            "MinusP_VBM_k": int(minus_edge["vbm_kpt_index"]),
            "MinusP_VBM_band": int(minus_edge["vbm_band_index"]),
            "MinusP_CBM_k": int(minus_edge["cbm_kpt_index"]),
            "MinusP_CBM_band": int(minus_edge["cbm_band_index"]),
        })

    rows.sort(key=lambda r: (DOPANT_ORDER.index(r["dopant"]), r["doping_percent"]))
    return data, rows


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Create 3x3 combined band figure and extract Eg(+P), Eg(-P), DeltaEg, and A_Eg."
    )
    parser.add_argument("--outdir", type=Path, default=Path("6combined_band_criticality_results"))
    parser.add_argument("--spin", type=int, default=1)
    parser.add_argument("--ymin", type=float, default=-0.6)
    parser.add_argument("--ymax", type=float, default=5.0)
    parser.add_argument("--klabels", type=str, default="Γ,X,Y,S,Z,U,T,R")
    parser.add_argument("--figure-name", type=str, default="combined_3x3_bands.png")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    klabels = [s.strip() for s in args.klabels.split(",")]

    print("Reading WFK files and extracting band edges...")
    data, rows = collect_all_data(spin=args.spin)

    print("Writing gap metric tables...")
    csv_path, txt_path = write_gap_metrics(rows, args.outdir)
    summary_path = write_criticality_summary(rows, args.outdir)

    print("Creating combined 3x3 band-structure figure...")
    fig_path = args.outdir / args.figure_name
    create_combined_3x3_figure(
        data=data,
        outpath=fig_path,
        ymin=args.ymin,
        ymax=args.ymax,
        klabels=klabels,
    )

    print("\nDone.")
    print(f"Figure:              {fig_path}")
    print(f"Gap metrics CSV:     {csv_path}")
    print(f"Gap metrics TXT:     {txt_path}")
    print(f"Criticality summary: {summary_path}")


if __name__ == "__main__":
    main()
