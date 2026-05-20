#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from netCDF4 import Dataset

"""
python3 1vbm_cbm_bands_delta_all.py 
"""

# ============================================================
# File list
# ============================================================
SYSTEMS: List[Dict[str, Any]] = [
    {
        "name": "3.125% Y-HfO2",
        "minus_label": "MinusP 3.125% Y-HfO2",
        "plus_label": "PlusP 3.125% Y-HfO2",
        "minus": Path(
            "/home/yanan/Material/Y_dope_project/3.125Per-Y-HfO2-2x2x2/"
            "berry-scf-results/berry_MinusP_3.125Per_Y_HfO2_12118864/"
            "berry_MinusP_3o_WFK.nc"
        ),
        "plus": Path(
            "/home/yanan/Material/Y_dope_project/3.125Per-Y-HfO2-2x2x2/"
            "berry-scf-results/berry_PlusP_3.125Per_Y_HfO2_12118862/"
            "berry_PlusP_3o_WFK.nc"
        ),
    },
    {
        "name": "6.25% Y-HfO2",
        "minus_label": "MinusP 6.25% Y-HfO2",
        "plus_label": "PlusP 6.25% Y-HfO2",
        "minus": Path(
            "/home/yanan/Material/Y_dope_project/6.25Per-Y-HfO2-2x2x2/"
            "berry-scf-results/berry_MinusP_6.25Y_HfO2_12397618/"
            "berry_MinusP_6o_WFK.nc"
        ),
        "plus": Path(
            "/home/yanan/Material/Y_dope_project/6.25Per-Y-HfO2-2x2x2/"
            "berry-scf-results/berry_PlusP_6.25Y_HfO2_12651723/"
            "berry_PlusP_6o_WFK.nc"
        ),
    },
        {
        "name": "9.375% Y-HfO2",
        "minus_label": "MinusP 9.375% Y-HfO2",
        "plus_label": "PlusP 9.375% Y-HfO2",
        "minus": Path(
            "/home/yanan/Material/Y_dope_project/9.375Y-HfO2-2x2x2/"
            "9.375Y-scf-berry-2x2x2-Results/berry_MinusP_9.375Y_HfO2_12121996/"
            "berry_MinusP_9o_WFK.nc"
        ),
        "plus": Path(
            "/home/yanan/Material/Y_dope_project/9.375Y-HfO2-2x2x2/"
            "9.375Y-scf-berry-2x2x2-Results/berry_PlusP_9.375Y_HfO2_12122002/"
            "berry_PlusP_9o_WFK.nc"
        ),
    },
    {
        "name": "3.125% Sc-HfO2",
        "minus_label": "MinusP 3.125% Sc-HfO2",
        "plus_label": "PlusP 3.125% Sc-HfO2",
        "minus": Path(
            "/home/yanan/Material/Y_dope_project/3.125Per-Sc-HfO2-2x2x2/"
            "berry-scf-Results/berry_MinusP_3.125Sc_HfO2_12144332/"
            "berry_MinusP_3o_WFK.nc"
        ),
        "plus": Path(
            "/home/yanan/Material/Y_dope_project/3.125Per-Sc-HfO2-2x2x2/"
            "berry-scf-Results/berry_PlusP_3.125Sc_HfO2_12144296/"
            "berry_PlusP_3o_WFK.nc"
        ),
    },
    {
        "name": "6.25% Sc-HfO2",
        "minus_label": "MinusP 6.25% Sc-HfO2",
        "plus_label": "PlusP 6.25% Sc-HfO2",
        "minus": Path(
            "/home/yanan/Material/Y_dope_project/6.25Per-Sc-HfO2-2x2x2/"
            "berry-scf-results/berry_MinusP_6.25Sc_HfO2_12144457/"
            "berry_MinusP_6o_WFK.nc"
        ),
        "plus": Path(
            "/home/yanan/Material/Y_dope_project/6.25Per-Sc-HfO2-2x2x2/"
            "berry-scf-results/berry_PlusP_6.25Sc_HfO2_12144349/"
            "berry_PlusP_6o_WFK.nc"
        ),
    },
    {
        "name": "9.375% Sc-HfO2",
        "minus_label": "MinusP 9.375% Sc-HfO2",
        "plus_label": "PlusP 9.375% Sc-HfO2",
        "minus": Path(
            "/home/yanan/Material/Y_dope_project/9.375Sc-HfO2-2x2x2/"
            "berry-scf-Results/berry_MinusP_9.375Sc_HfO2_12144529/"
            "berry_MinusP_9o_WFK.nc"
        ),
        "plus": Path(
            "/home/yanan/Material/Y_dope_project/9.375Sc-HfO2-2x2x2/"
            "berry-scf-Results/berry_PlusP_9.375Sc_HfO2_12144507/"
            "berry_PlusP_9o_WFK.nc"
        ),
    },
     {
        "name": "3.125% Zr-HfO2",
        "minus_label": "MinusP 3.125% Zr-HfO2",
        "plus_label": "PlusP 3.125% Zr-HfO2",
        "minus": Path(
            "/home/yanan/Material/Y_dope_project/3.125Per-Zr-HfO2-2x2x2/"
            "berry-scf-Results/berry_MinusP_3.125Zr_HfO2_12247642/"
            "berry_MinusP_3o_WFK.nc"
        ),
        "plus": Path(
            "/home/yanan/Material/Y_dope_project/3.125Per-Zr-HfO2-2x2x2/"
            "berry-scf-Results/berry_PlusP_3.125Zr_HfO2_12247641/"
            "berry_PlusP_3o_WFK.nc"
        ),
    },   
        {
        "name": "6.25% Zr-HfO2",
        "minus_label": "MinusP 6.25% Zr-HfO2",
        "plus_label": "PlusP 6.25% Zr-HfO2",
        "minus": Path(
            "/home/yanan/Material/Y_dope_project/6.25Per-Zr-HfO2-2x2x2/"
            "berry-scf-Results/berry_MinusP_6.25Zr_HfO2_12248602/"
            "berry_MinusP_6o_WFK.nc"
        ),
        "plus": Path(
            "/home/yanan/Material/Y_dope_project/6.25Per-Zr-HfO2-2x2x2/"
            "berry-scf-Results/berry_PlusP_6.25Zr_HfO2_12248413/"
            "berry_PlusP_6o_WFK.nc"
        ),
    },
         {
        "name": "9.375% Zr-HfO2",
        "minus_label": "MinusP 9.375% Zr-HfO2",
        "plus_label": "PlusP 9.375% Zr-HfO2",
        "minus": Path(
            "/home/yanan/Material/Y_dope_project/9.375Per-Zr-HfO2-2x2x2/"
            "berry-scf-Results/berry_MinusP_9.375Zr_HfO2_12247637/"
            "berry_MinusP_9o_WFK.nc"
        ),
        "plus": Path(
            "/home/yanan/Material/Y_dope_project/9.375Per-Zr-HfO2-2x2x2/"
            "berry-scf-Results/berry_PlusP_9.375Zr_HfO2_12247629/"
            "berry_PlusP_9o_WFK.nc"
        ),
    },
]


# ============================================================
# Helpers
# ============================================================
def find_var(ds: Dataset, candidates: List[str]) -> str:
    for name in candidates:
        if name in ds.variables:
            return name
    raise KeyError(
        f"None of these variables were found: {candidates}\n"
        f"Available variables: {list(ds.variables.keys())}"
    )


def normalize_eig_occ_shapes(eig: np.ndarray, occ: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    eig = np.squeeze(eig)
    occ = np.squeeze(occ)

    if eig.ndim == 3:
        # Common cases:
        #   (nsppol, nkpt, nband)
        #   (nkpt, nband, nsppol)
        if eig.shape[0] <= 2:
            eig = eig[0]
            occ = occ[0]
        elif eig.shape[-1] <= 2:
            eig = eig[..., 0]
            occ = occ[..., 0]
        else:
            raise ValueError(f"Cannot infer eig shape: {eig.shape}")

    if occ.ndim == 3:
        if occ.shape[0] <= 2:
            occ = occ[0]
        elif occ.shape[-1] <= 2:
            occ = occ[..., 0]
        else:
            raise ValueError(f"Cannot infer occ shape: {occ.shape}")

    if eig.ndim != 2 or occ.ndim != 2:
        raise ValueError(f"Unexpected shapes after normalization: eig={eig.shape}, occ={occ.shape}")

    if eig.shape != occ.shape:
        raise ValueError(f"eig/occ shape mismatch: eig={eig.shape}, occ={occ.shape}")

    return eig, occ


def get_kpoints(ds: Dataset, nkpt: int) -> np.ndarray | None:
    kpt_candidates = [
        "reduced_coordinates_of_kpoints",
        "kpt",
        "kptns",
        "reduced_coordinates",
    ]
    for name in kpt_candidates:
        if name in ds.variables:
            arr = np.array(ds.variables[name][:]).squeeze()
            if arr.ndim == 2 and arr.shape[0] == nkpt and arr.shape[1] == 3:
                return arr
            if arr.ndim == 2 and arr.shape[1] == nkpt and arr.shape[0] == 3:
                return arr.T
    return None


def analyze_wfk(path: Path, label: str) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label}: file not found:\n{path}")

    with Dataset(path, "r") as ds:
        eig_name = find_var(ds, ["eigenvalues", "eig", "Eigenvalues"])
        occ_name = find_var(ds, ["occupations", "occ", "Occupations"])

        eig_raw = np.array(ds.variables[eig_name][:])
        occ_raw = np.array(ds.variables[occ_name][:])

        eig_raw = np.squeeze(eig_raw)
        occ_raw = np.squeeze(occ_raw)

        # Force shape to (nsppol, nkpt, nband)
        if eig_raw.ndim == 3:
            if eig_raw.shape[0] <= 2:
                eig = eig_raw
                occ = occ_raw
            elif eig_raw.shape[-1] <= 2:
                eig = np.transpose(eig_raw, (2, 0, 1))
                occ = np.transpose(occ_raw, (2, 0, 1))
            else:
                raise ValueError(f"Cannot infer eig shape: {eig_raw.shape}")
        elif eig_raw.ndim == 2:
            eig = eig_raw[np.newaxis, :, :]
            occ = occ_raw[np.newaxis, :, :]
        else:
            raise ValueError(f"Unexpected eig shape: {eig_raw.shape}")

        nsppol, nkpt, nband = eig.shape
        kpts = get_kpoints(ds, nkpt)

    occ_tol_occ = 0.5
    occ_tol_emp = 0.5

    vbm_e = -1.0e99
    cbm_e =  1.0e99
    vbm_s = vbm_k = vbm_b = None
    cbm_s = cbm_k = cbm_b = None

    for isppol in range(nsppol):
        for ik in range(nkpt):
            for ib in range(nband):
                e = float(eig[isppol, ik, ib])
                o = float(occ[isppol, ik, ib])

                if o > occ_tol_occ and e > vbm_e:
                    vbm_e = e
                    vbm_s = isppol + 1
                    vbm_k = ik + 1
                    vbm_b = ib + 1

                if o <= occ_tol_emp and e < cbm_e:
                    cbm_e = e
                    cbm_s = isppol + 1
                    cbm_k = ik + 1
                    cbm_b = ib + 1

    if None in (vbm_s, vbm_k, vbm_b, cbm_s, cbm_k, cbm_b):
        raise RuntimeError(f"{label}: could not identify VBM/CBM from occupations.")

    result = {
        "label": label,
        "path": str(path),
        "nsppol": nsppol,
        "nkpt": nkpt,
        "nband": nband,
        "vbm_energy_ev": vbm_e,
        "vbm_spin_index": vbm_s,
        "vbm_kpt_index": vbm_k,
        "vbm_band_index": vbm_b,
        "cbm_energy_ev": cbm_e,
        "cbm_spin_index": cbm_s,
        "cbm_kpt_index": cbm_k,
        "cbm_band_index": cbm_b,
        "gap_ev": cbm_e - vbm_e,
    }

    if kpts is not None:
        result["vbm_kpt_coords"] = kpts[vbm_k - 1]
        result["cbm_kpt_coords"] = kpts[cbm_k - 1]

    return result

def fmt_kpt(arr: np.ndarray | None) -> str:
    if arr is None:
        return "N/A"
    return f"({arr[0]:.6f}, {arr[1]:.6f}, {arr[2]:.6f})"


# ============================================================
# Printing
# ============================================================
def print_result(res: Dict[str, Any]) -> None:
    print("=" * 90)
    print(f"{res['label']}")
    print("=" * 90)
    print(f"File   : {res['path']}")
    print(f"nkpt   : {res['nkpt']}")
    print(f"nband  : {res['nband']}")
    print()
    print(f"VBM    : {res['vbm_energy_ev']:.6f} eV")
    print(f"         k-point index = {res['vbm_kpt_index']}")
    print(f"         band index    = {res['vbm_band_index']}")
    print(f"         k-point coord = {fmt_kpt(res.get('vbm_kpt_coords'))}")
    print()
    print(f"CBM    : {res['cbm_energy_ev']:.6f} eV")
    print(f"         k-point index = {res['cbm_kpt_index']}")
    print(f"         band index    = {res['cbm_band_index']}")
    print(f"         k-point coord = {fmt_kpt(res.get('cbm_kpt_coords'))}")
    print()
    print(f"Gap    : {res['gap_ev']:.6f} eV")
    print()


def print_pair_comparison(system_name: str, minus_res: Dict[str, Any], plus_res: Dict[str, Any]) -> None:
    print("=" * 90)
    print(f"COMPARISON: {system_name}  (MinusP vs PlusP)")
    print("=" * 90)
    print(f"{'Quantity':<24} {'MinusP':>20} {'PlusP':>20}")
    print("-" * 90)
    print(f"{'VBM energy (eV)':<24} {minus_res['vbm_energy_ev']:>20.6f} {plus_res['vbm_energy_ev']:>20.6f}")
    print(f"{'CBM energy (eV)':<24} {minus_res['cbm_energy_ev']:>20.6f} {plus_res['cbm_energy_ev']:>20.6f}")
    print(f"{'Gap (eV)':<24} {minus_res['gap_ev']:>20.6f} {plus_res['gap_ev']:>20.6f}")
    print(f"{'VBM kpt/band':<24} "
          f"{str((minus_res['vbm_kpt_index'], minus_res['vbm_band_index'])):>20} "
          f"{str((plus_res['vbm_kpt_index'], plus_res['vbm_band_index'])):>20}")
    print(f"{'CBM kpt/band':<24} "
          f"{str((minus_res['cbm_kpt_index'], minus_res['cbm_band_index'])):>20} "
          f"{str((plus_res['cbm_kpt_index'], plus_res['cbm_band_index'])):>20}")
    print("-" * 90)
    print(f"{'ΔVBM (Plus-Minus)':<24} {(plus_res['vbm_energy_ev'] - minus_res['vbm_energy_ev']):>20.6f}")
    print(f"{'ΔCBM (Plus-Minus)':<24} {(plus_res['cbm_energy_ev'] - minus_res['cbm_energy_ev']):>20.6f}")
    print(f"{'ΔGap (Plus-Minus)':<24} {(plus_res['gap_ev'] - minus_res['gap_ev']):>20.6f}")
    print("=" * 90)
    print()


def print_global_summary(rows: List[Dict[str, Any]]) -> None:
    print("=" * 130)
    print("GLOBAL SUMMARY")
    print("=" * 130)
    header = (
        f"{'System':<20} {'State':<8} "
        f"{'VBM (eV)':>12} {'VBM k/b':>14} "
        f"{'CBM (eV)':>12} {'CBM k/b':>14} "
        f"{'Gap (eV)':>12}"
    )
    print(header)
    print("-" * 130)

    for row in rows:
        print(
            f"{row['system']:<20} {row['state']:<8} "
            f"{row['vbm_energy_ev']:>12.6f} "
            f"{str((row['vbm_kpt_index'], row['vbm_band_index'])):>14} "
            f"{row['cbm_energy_ev']:>12.6f} "
            f"{str((row['cbm_kpt_index'], row['cbm_band_index'])):>14} "
            f"{row['gap_ev']:>12.6f}"
        )

    print("=" * 130)


# ============================================================
# Main
# ============================================================
def main() -> None:
    summary_rows: List[Dict[str, Any]] = []

    # ✅ Create folder
    output_dir = Path("1vbm_cbm_bands_all_Results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ✅ Output file inside folder
    output_file = output_dir / "1vbm_cbm_bands_all_results.txt"

    with open(output_file, "w") as f:

        def write(line=""):
            print(line)
            f.write(line + "\n")

        for system in SYSTEMS:
            minus_res = analyze_wfk(system["minus"], system["minus_label"])
            plus_res = analyze_wfk(system["plus"], system["plus_label"])

            # ---------- Individual results ----------
            for res in [minus_res, plus_res]:
                write("=" * 90)
                write(f"{res['label']}")
                write("=" * 90)
                write(f"File   : {res['path']}")
                write(f"nkpt   : {res['nkpt']}")
                write(f"nband  : {res['nband']}")
                write()

                write(f"VBM    : {res['vbm_energy_ev']:.6f} eV")
                write(f"         k-point index = {res['vbm_kpt_index']}")
                write(f"         band index    = {res['vbm_band_index']}")
                write(f"         k-point coord = {fmt_kpt(res.get('vbm_kpt_coords'))}")
                write()

                write(f"CBM    : {res['cbm_energy_ev']:.6f} eV")
                write(f"         k-point index = {res['cbm_kpt_index']}")
                write(f"         band index    = {res['cbm_band_index']}")
                write(f"         k-point coord = {fmt_kpt(res.get('cbm_kpt_coords'))}")
                write()

                write(f"Gap    : {res['gap_ev']:.6f} eV")
                write()

            # ---------- Comparison ----------
            write("=" * 90)
            write(f"COMPARISON: {system['name']} (MinusP vs PlusP)")
            write("=" * 90)
            write(f"{'Quantity':<24} {'MinusP':>20} {'PlusP':>20}")
            write("-" * 90)

            write(f"{'VBM energy (eV)':<24} {minus_res['vbm_energy_ev']:>20.6f} {plus_res['vbm_energy_ev']:>20.6f}")
            write(f"{'CBM energy (eV)':<24} {minus_res['cbm_energy_ev']:>20.6f} {plus_res['cbm_energy_ev']:>20.6f}")
            write(f"{'Gap (eV)':<24} {minus_res['gap_ev']:>20.6f} {plus_res['gap_ev']:>20.6f}")

            write(f"{'VBM kpt/band':<24} "
                  f"{str((minus_res['vbm_kpt_index'], minus_res['vbm_band_index'])):>20} "
                  f"{str((plus_res['vbm_kpt_index'], plus_res['vbm_band_index'])):>20}")

            write(f"{'CBM kpt/band':<24} "
                  f"{str((minus_res['cbm_kpt_index'], minus_res['cbm_band_index'])):>20} "
                  f"{str((plus_res['cbm_kpt_index'], plus_res['cbm_band_index'])):>20}")

            write("-" * 90)
            write(f"{'ΔVBM (Plus-Minus)':<24} {(plus_res['vbm_energy_ev'] - minus_res['vbm_energy_ev']):>20.6f}")
            write(f"{'ΔCBM (Plus-Minus)':<24} {(plus_res['cbm_energy_ev'] - minus_res['cbm_energy_ev']):>20.6f}")
            write(f"{'ΔGap (Plus-Minus)':<24} {(plus_res['gap_ev'] - minus_res['gap_ev']):>20.6f}")
            write("=" * 90)
            write()

            summary_rows.append({"system": system["name"], "state": "MinusP", **minus_res})
            summary_rows.append({"system": system["name"], "state": "PlusP", **plus_res})

        # ---------- Global summary ----------
        write("=" * 130)
        write("GLOBAL SUMMARY")
        write("=" * 130)

        header = (
            f"{'System':<20} {'State':<8} "
            f"{'VBM (eV)':>12} {'VBM k/b':>14} "
            f"{'CBM (eV)':>12} {'CBM k/b':>14} "
            f"{'Gap (eV)':>12}"
        )
        write(header)
        write("-" * 130)

        for row in summary_rows:
            write(
                f"{row['system']:<20} {row['state']:<8} "
                f"{row['vbm_energy_ev']:>12.6f} "
                f"{str((row['vbm_kpt_index'], row['vbm_band_index'])):>14} "
                f"{row['cbm_energy_ev']:>12.6f} "
                f"{str((row['cbm_kpt_index'], row['cbm_band_index'])):>14} "
                f"{row['gap_ev']:>12.6f}"
            )

        write("=" * 130)

    print(f"\n✅ Results saved to: {output_file}")
    
if __name__ == "__main__":
    main()
