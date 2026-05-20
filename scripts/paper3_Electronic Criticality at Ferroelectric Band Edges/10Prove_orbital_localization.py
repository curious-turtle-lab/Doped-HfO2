#!/usr/bin/env python3
"""
10_prove_orbital_localization_no_labels.py

Reads cube files from:
  /home/yanan/Material/Y_dope_project/CBM_VBM_Sc/9Cube_WFK_ALL_Best_Figures

Generates output folder:
  10Prove_orbital_localization

Main figure added:
  Fig_spectral_function_style_localization_no_labels.png
  Fig_spectral_function_style_localization_no_labels.pdf

This version removes crowded peak labels and explains all cases using the heatmap.
"""

import os
import re
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

ROOT_DIR = "/home/yanan/Material/Y_dope_project/CBM_VBM_Sc/9Cube_WFK_ALL_Best_Figures"
OUT_DIR = os.path.join(ROOT_DIR, "10Prove_orbital_localization")

TOP_PERCENT = 1.0
DOPANT_RADIUS_ANG = 3.0
ATOM_NEIGHBOR_RADIUS_ANG = 2.5

DOPANTS = ["Sc", "Y", "Zr"]

DOPING_FOLDER_MAP = {
    "3.125": "3p125",
    "6.25": "6p25",
    "9.375": "9p375",
}

CODE_TO_PERCENT = {
    "3p125": "3.125",
    "6p25": "6.25",
    "9p375": "9.375",
}

MAKE_PLOTS = True


# ============================================================
# FOLDER INVENTORY
# ============================================================

def collect_folder_inventory(root_dir):
    records = []

    folders = sorted([
        f for f in glob.glob(os.path.join(root_dir, "*"))
        if os.path.isdir(f)
        and os.path.basename(f) != "10Prove_orbital_localization"
    ])

    for folder in folders:
        folder_name = os.path.basename(folder)
        files = sorted(glob.glob(os.path.join(folder, "*")))

        for f in files:
            ext = os.path.splitext(f)[1].lower()

            records.append({
                "root_dir": root_dir,
                "folder_name": folder_name,
                "folder_path": folder,
                "file_name": os.path.basename(f),
                "file_path": f,
                "extension": ext,
                "is_cube": ext == ".cube",
                "is_png": ext == ".png",
                "is_vesta": ext == ".vesta",
                "file_size_MB": os.path.getsize(f) / 1024**2,
            })

    return pd.DataFrame(records)


def summarize_folders(df_inventory):
    if df_inventory.empty:
        return pd.DataFrame()

    return (
        df_inventory
        .groupby(["folder_name", "folder_path"])
        .agg(
            total_files=("file_name", "count"),
            cube_files=("is_cube", "sum"),
            png_files=("is_png", "sum"),
            vesta_files=("is_vesta", "sum"),
            total_size_MB=("file_size_MB", "sum"),
        )
        .reset_index()
    )


# ============================================================
# PARSING
# ============================================================

def parse_folder_name(folder_name):
    m = re.match(r"([0-9.]+)(Sc|Y|Zr)_WFK_Best", folder_name)

    if not m:
        return None, None, None

    doping_percent = m.group(1)
    dopant = m.group(2)
    doping_code = DOPING_FOLDER_MAP.get(doping_percent)

    return doping_percent, doping_code, dopant


def parse_cube_name(path):
    base = os.path.basename(path)
    folder_path = os.path.dirname(path)
    folder_name = os.path.basename(folder_path)

    folder_doping_percent, folder_doping_code, folder_dopant = parse_folder_name(folder_name)

    if "PlusP" in base:
        polarization = "+P"
    elif "MinusP" in base:
        polarization = "-P"
    else:
        polarization = None

    if "VBM" in base.upper():
        orbital = "VBM"
    elif "CBM" in base.upper():
        orbital = "CBM"
    else:
        orbital = None

    dopant = folder_dopant

    for d in DOPANTS:
        if d in base:
            dopant = d

    doping_code = folder_doping_code
    doping_percent = folder_doping_percent

    m_doping = re.search(r"(3p125|6p25|9p375)", base)
    if m_doping:
        doping_code = m_doping.group(1)
        doping_percent = CODE_TO_PERCENT.get(doping_code, doping_percent)

    k_match = re.search(r"_k(\d+)", base)
    b_match = re.search(r"_b(\d+)", base)

    kpt = int(k_match.group(1)) if k_match else None
    band_index = int(b_match.group(1)) if b_match else None

    return {
        "folder_name": folder_name,
        "folder_path": folder_path,
        "file": path,
        "file_name": base,
        "doping_percent": doping_percent,
        "doping_code": doping_code,
        "dopant": dopant,
        "polarization": polarization,
        "orbital": orbital,
        "kpt": kpt,
        "band_index": band_index,
    }


# ============================================================
# CUBE READER
# ============================================================

def read_cube(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    natoms_origin = lines[2].split()
    natoms = int(float(natoms_origin[0]))
    origin = np.array([float(x) for x in natoms_origin[1:4]])

    grid = []
    axes = []

    for i in range(3):
        parts = lines[3 + i].split()
        n = int(float(parts[0]))
        vec = np.array([float(x) for x in parts[1:4]])
        grid.append(n)
        axes.append(vec)

    axes = np.array(axes)
    voxel_volume = abs(np.linalg.det(axes))

    atoms = []
    atom_start = 6

    for i in range(abs(natoms)):
        parts = lines[atom_start + i].split()
        Z = int(float(parts[0]))
        charge = float(parts[1])
        xyz = np.array([float(x) for x in parts[2:5]])
        atoms.append({
            "Z": Z,
            "charge": charge,
            "xyz": xyz,
        })

    data_start = atom_start + abs(natoms)

    values = []
    for line in lines[data_start:]:
        values.extend([float(x) for x in line.split()])

    expected = grid[0] * grid[1] * grid[2]

    if len(values) < expected:
        raise ValueError(
            f"Cube data too short: expected {expected}, got {len(values)}"
        )

    values = values[:expected]
    data = np.array(values).reshape(tuple(grid))

    return atoms, origin, axes, data, voxel_volume


# ============================================================
# LOCALIZATION METRICS
# ============================================================

def make_coordinates(shape, origin, axes):
    ix, iy, iz = np.indices(shape)

    coords = (
        origin[None, None, None, :]
        + ix[..., None] * axes[0]
        + iy[..., None] * axes[1]
        + iz[..., None] * axes[2]
    )

    return coords


def orbital_metrics(cube_file):
    atoms, origin, axes, psi, voxel_volume = read_cube(cube_file)

    rho = np.abs(psi) ** 2
    total = rho.sum()

    if total <= 0:
        raise ValueError(f"Zero orbital norm in {cube_file}")

    rho_norm = rho / total

    ipr = np.sum(rho_norm ** 2)
    pr_voxels = 1.0 / ipr
    total_voxels = rho.size
    pr_fraction = pr_voxels / total_voxels
    v_eff = pr_voxels * voxel_volume

    flat = rho_norm.ravel()
    cutoff_index = max(1, int(len(flat) * TOP_PERCENT / 100.0))
    top_weight = np.sort(flat)[::-1][:cutoff_index].sum()

    coords = make_coordinates(rho.shape, origin, axes)
    com = np.sum(coords * rho_norm[..., None], axis=(0, 1, 2))

    atom_weight_sum = 0.0

    for atom in atoms:
        dist = np.linalg.norm(coords - atom["xyz"], axis=-1)
        atom_weight_sum += rho_norm[dist <= ATOM_NEIGHBOR_RADIUS_ANG].sum()

    dopant_Z = {
        "Sc": 21,
        "Y": 39,
        "Zr": 40,
    }

    dopant_positions = []

    for atom in atoms:
        if atom["Z"] in dopant_Z.values():
            dopant_positions.append(atom["xyz"])

    dopant_weight = np.nan

    if len(dopant_positions) > 0:
        mask = np.zeros(rho.shape, dtype=bool)

        for pos in dopant_positions:
            dist = np.linalg.norm(coords - pos, axis=-1)
            mask |= dist <= DOPANT_RADIUS_ANG

        dopant_weight = rho_norm[mask].sum()

    return {
        "IPR": ipr,
        "PR_voxels": pr_voxels,
        "PR_fraction": pr_fraction,
        "Veff": v_eff,
        f"Top_{TOP_PERCENT:.1f}pct_weight": top_weight,
        "COM_x": com[0],
        "COM_y": com[1],
        "COM_z": com[2],
        f"Atom_weight_R{ATOM_NEIGHBOR_RADIUS_ANG}A": atom_weight_sum,
        f"Dopant_weight_R{DOPANT_RADIUS_ANG}A": dopant_weight,
        "voxel_volume": voxel_volume,
        "nvox": total_voxels,
        "natoms": len(atoms),
    }


def orbital_overlap(cube_vbm, cube_cbm):
    _, _, _, psi1, voxel_volume1 = read_cube(cube_vbm)
    _, _, _, psi2, voxel_volume2 = read_cube(cube_cbm)

    if psi1.shape != psi2.shape:
        raise ValueError("VBM and CBM cube grids do not match.")

    rho1 = np.abs(psi1) ** 2
    rho2 = np.abs(psi2) ** 2

    numerator = np.sum(rho1 * rho2) * voxel_volume1

    denominator = math.sqrt(
        np.sum(rho1 ** 2) * voxel_volume1 *
        np.sum(rho2 ** 2) * voxel_volume2
    )

    if denominator == 0:
        return np.nan

    return numerator / denominator


# ============================================================
# BASIC PLOTS
# ============================================================

def make_label(row):
    return (
        f"{row['doping_percent']}% {row['dopant']} "
        f"{row['polarization']} {row['orbital']}"
    )


def plot_bar(df, ycol, ylabel, title, output_name):
    d = df.copy()
    d["label"] = d.apply(make_label, axis=1)

    plt.figure(figsize=(16, 5))
    plt.bar(np.arange(len(d)), d[ycol])
    plt.xticks(np.arange(len(d)), d["label"], rotation=90, fontsize=7)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, output_name), dpi=300)
    plt.close()


def plot_overlap(df_overlap):
    if df_overlap.empty:
        return

    d = df_overlap.copy()

    d["label"] = (
        d["doping_percent"].astype(str)
        + "% "
        + d["dopant"].astype(str)
        + " "
        + d["polarization"].astype(str)
    )

    plt.figure(figsize=(12, 5))
    plt.bar(np.arange(len(d)), d["VBM_CBM_overlap"])
    plt.xticks(np.arange(len(d)), d["label"], rotation=90, fontsize=8)
    plt.ylabel("Normalized VBM-CBM density overlap")
    plt.title("VBM-CBM spatial overlap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Fig_VBM_CBM_overlap_proof.png"), dpi=300)
    plt.close()


# ============================================================
# CLEAN SPECTRAL-FUNCTION-STYLE FIGURE WITHOUT PEAK LABELS
# ============================================================

def plot_spectral_function_style_localization_no_labels(df):
    """
    Clean spectral-function-style figure.

    Panel (a):
        Gaussian-broadened VBM/CBM localization intensity from IPR.
        No individual peak labels.

    Panel (b):
        Heatmap for all dopant/concentration/polarization cases.
        This is the main explanation of all cases.
    """

    d = df.copy()

    d = d.sort_values(
        ["dopant", "doping_percent", "polarization", "orbital", "band_index"]
    ).reset_index(drop=True)

    # Energy-like coordinate from band index.
    # Replace this with real eigenvalue_eV if available later.
    mean_band = d["band_index"].mean()
    d["energy_proxy"] = (d["band_index"] - mean_band) * 0.25

    # Normalize IPR
    d["IPR_norm"] = d["IPR"] / d["IPR"].max()

    d["case_label"] = (
        d["doping_percent"].astype(str)
        + "% "
        + d["dopant"].astype(str)
        + " "
        + d["polarization"].astype(str)
    )

    Emin = d["energy_proxy"].min() - 1.0
    Emax = d["energy_proxy"].max() + 1.0
    Egrid = np.linspace(Emin, Emax, 900)

    sigma = 0.08

    def gaussian(E, E0, amp, sig):
        return amp * np.exp(-0.5 * ((E - E0) / sig) ** 2)

    vbm = d[d["orbital"] == "VBM"].copy()
    cbm = d[d["orbital"] == "CBM"].copy()

    A_vbm = np.zeros_like(Egrid)
    A_cbm = np.zeros_like(Egrid)

    for _, row in vbm.iterrows():
        A_vbm += gaussian(Egrid, row["energy_proxy"], row["IPR_norm"], sigma)

    for _, row in cbm.iterrows():
        A_cbm += gaussian(Egrid, row["energy_proxy"], row["IPR_norm"], sigma)

    if A_vbm.max() > 0:
        A_vbm = A_vbm / A_vbm.max()

    if A_cbm.max() > 0:
        A_cbm = A_cbm / A_cbm.max()

    # All cases for heatmap
    unique_cases = (
        d[["dopant", "doping_percent", "polarization", "case_label"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Better ordering: Sc, Y, Zr and low-to-high doping, +P then -P
    dopant_order = {"Sc": 0, "Y": 1, "Zr": 2}
    pol_order = {"+P": 0, "-P": 1}

    unique_cases["dopant_order"] = unique_cases["dopant"].map(dopant_order)
    unique_cases["doping_float"] = unique_cases["doping_percent"].astype(float)
    unique_cases["pol_order"] = unique_cases["polarization"].map(pol_order)

    unique_cases = unique_cases.sort_values(
        ["dopant_order", "doping_float", "pol_order"]
    ).reset_index(drop=True)

    Amap = np.zeros((len(unique_cases), len(Egrid)))

    for i, case in unique_cases.iterrows():
        sub = d[
            (d["dopant"] == case["dopant"])
            & (d["doping_percent"] == case["doping_percent"])
            & (d["polarization"] == case["polarization"])
        ]

        for _, row in sub.iterrows():
            Amap[i, :] += gaussian(
                Egrid,
                row["energy_proxy"],
                row["IPR_norm"],
                sigma,
            )

    if Amap.max() > 0:
        Amap = Amap / Amap.max()

    # ========================================================
    # Plot
    # ========================================================

    fig = plt.figure(figsize=(10, 11))

    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.45],
        hspace=0.45
    )

    # -------------------------
    # Panel (a)
    # -------------------------

    ax1 = fig.add_subplot(gs[0, 0])

    ax1.fill_between(Egrid, A_vbm, alpha=0.50, label="VBM")
    ax1.plot(Egrid, A_vbm, linewidth=2.0)

    ax1.fill_between(Egrid, A_cbm, alpha=0.45, label="CBM")
    ax1.plot(Egrid, A_cbm, linewidth=2.0)

    ax1.axvline(0.0, linestyle="--", linewidth=1.2)

    ax1.text(
        0.02,
        0.88,
        "(a)",
        transform=ax1.transAxes,
        fontsize=18,
        fontweight="bold"
    )

    ax1.set_xlabel("Energy-like coordinate from band index", fontsize=13)
    ax1.set_ylabel(r"$A_{\mathrm{loc}}(E)$", fontsize=15)
    ax1.set_title("Localization spectral intensity", fontsize=14, pad=16)
    ax1.legend(frameon=False, fontsize=11, loc="upper right")

    ax1.set_ylim(0, 1.15)

    # -------------------------
    # Panel (b)
    # -------------------------

    ax2 = fig.add_subplot(gs[1, 0])

    extent = [
        Egrid.min(),
        Egrid.max(),
        0,
        len(unique_cases)
    ]

    im = ax2.imshow(
        Amap,
        aspect="auto",
        origin="lower",
        extent=extent,
        interpolation="nearest"
    )

    ax2.text(
        0.02,
        0.90,
        "(b)",
        transform=ax2.transAxes,
        fontsize=18,
        fontweight="bold",
        color="white"
    )

    ax2.axvline(0.0, linestyle="--", linewidth=1.2)

    ax2.set_xlabel("Energy-like coordinate from band index", fontsize=13)
    ax2.set_ylabel("Doped HfO$_2$ case", fontsize=13)

    ax2.set_yticks(np.arange(len(unique_cases)) + 0.5)
    ax2.set_yticklabels(unique_cases["case_label"], fontsize=8)

    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label(r"Normalized localization weight $Z_{\mathrm{loc}}$", fontsize=12)

    fig.suptitle(
        "Frontier-state localization spectral map",
        fontsize=16,
        y=0.985
    )

    out_png = os.path.join(
        OUT_DIR,
        "Fig_spectral_function_style_localization_no_labels.png"
    )

    out_pdf = os.path.join(
        OUT_DIR,
        "Fig_spectral_function_style_localization_no_labels.pdf"
    )

    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print("Saved clean spectral-function-style figure:")
    print(out_png)
    print(out_pdf)


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 80)
    print("PROVE ORBITAL LOCALIZATION")
    print("=" * 80)
    print("Input root folder:")
    print(ROOT_DIR)
    print("\nOutput folder:")
    print(OUT_DIR)

    # ------------------------------------------------------------
    # Folder inventory
    # ------------------------------------------------------------

    df_inventory = collect_folder_inventory(ROOT_DIR)

    inventory_path = os.path.join(OUT_DIR, "folder_file_inventory.csv")
    df_inventory.to_csv(inventory_path, index=False)

    df_folder_summary = summarize_folders(df_inventory)
    folder_summary_path = os.path.join(OUT_DIR, "folder_summary.csv")
    df_folder_summary.to_csv(folder_summary_path, index=False)

    # ------------------------------------------------------------
    # Find cube files
    # ------------------------------------------------------------

    cube_files = sorted(glob.glob(os.path.join(ROOT_DIR, "*", "*.cube")))

    cube_files = [
        f for f in cube_files
        if "10Prove_orbital_localization" not in f
    ]

    print("\nTotal cube files found:", len(cube_files))

    if len(cube_files) == 0:
        raise RuntimeError("No cube files found. Check ROOT_DIR.")

    # ------------------------------------------------------------
    # Compute localization metrics
    # ------------------------------------------------------------

    records = []

    for cube_file in cube_files:
        info = parse_cube_name(cube_file)

        if info["polarization"] is None or info["orbital"] is None:
            print("Skipping unrecognized cube:", cube_file)
            continue

        print("\nReading:")
        print("  folder =", info["folder_name"])
        print("  file   =", info["file_name"])

        try:
            metrics = orbital_metrics(cube_file)
        except Exception as e:
            print("  ERROR:", e)
            continue

        row = {}
        row.update(info)
        row.update(metrics)
        records.append(row)

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError("No valid localization metrics generated.")

    df = df.sort_values(
        ["dopant", "doping_percent", "polarization", "orbital", "kpt", "band_index"]
    )

    metrics_path = os.path.join(OUT_DIR, "orbital_localization_metrics.csv")
    df.to_csv(metrics_path, index=False)

    # ------------------------------------------------------------
    # VBM-CBM overlap
    # ------------------------------------------------------------

    overlap_records = []

    group_cols = [
        "folder_name",
        "folder_path",
        "dopant",
        "doping_percent",
        "doping_code",
        "polarization",
    ]

    for key, g in df.groupby(group_cols, dropna=False):
        folder_name, folder_path, dopant, doping_percent, doping_code, pol = key

        vbm = g[g["orbital"] == "VBM"]
        cbm = g[g["orbital"] == "CBM"]

        if len(vbm) == 0 or len(cbm) == 0:
            continue

        vbm_file = vbm.iloc[0]["file"]
        cbm_file = cbm.iloc[0]["file"]

        try:
            overlap = orbital_overlap(vbm_file, cbm_file)
        except Exception as e:
            print("Overlap failed:", folder_name, pol, e)
            overlap = np.nan

        overlap_records.append({
            "folder_name": folder_name,
            "folder_path": folder_path,
            "dopant": dopant,
            "doping_percent": doping_percent,
            "doping_code": doping_code,
            "polarization": pol,
            "VBM_file": os.path.basename(vbm_file),
            "CBM_file": os.path.basename(cbm_file),
            "VBM_CBM_overlap": overlap,
        })

    df_overlap = pd.DataFrame(overlap_records)

    overlap_path = os.path.join(OUT_DIR, "vbm_cbm_overlap_metrics.csv")
    df_overlap.to_csv(overlap_path, index=False)

    # ------------------------------------------------------------
    # Publication summary
    # ------------------------------------------------------------

    summary_cols = [
        "folder_name",
        "file_name",
        "dopant",
        "doping_percent",
        "doping_code",
        "polarization",
        "orbital",
        "kpt",
        "band_index",
        "IPR",
        "PR_voxels",
        "PR_fraction",
        f"Top_{TOP_PERCENT:.1f}pct_weight",
        f"Dopant_weight_R{DOPANT_RADIUS_ANG}A",
        "Veff",
        "COM_x",
        "COM_y",
        "COM_z",
        "natoms",
        "nvox",
    ]

    df_summary = df[summary_cols]

    summary_path = os.path.join(OUT_DIR, "publication_summary_localization_proof.csv")
    df_summary.to_csv(summary_path, index=False)

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------

    if MAKE_PLOTS:
        plot_bar(
            df,
            "IPR",
            "IPR",
            "Inverse Participation Ratio: larger IPR means stronger localization",
            "Fig_IPR_localization_proof.png",
        )

        plot_bar(
            df,
            "PR_fraction",
            "Participation fraction",
            "Participation fraction: smaller value means stronger localization",
            "Fig_PR_fraction_localization_proof.png",
        )

        plot_bar(
            df,
            f"Top_{TOP_PERCENT:.1f}pct_weight",
            f"Weight in top {TOP_PERCENT:.1f}% density voxels",
            "Density concentration proof",
            "Fig_top_density_weight_proof.png",
        )

        plot_bar(
            df,
            f"Dopant_weight_R{DOPANT_RADIUS_ANG}A",
            f"Orbital weight within {DOPANT_RADIUS_ANG} Å of dopant",
            "Dopant-neighborhood orbital localization",
            "Fig_dopant_weight_proof.png",
        )

        plot_overlap(df_overlap)

        plot_spectral_function_style_localization_no_labels(df)

    # ------------------------------------------------------------
    # README
    # ------------------------------------------------------------

    readme_path = os.path.join(OUT_DIR, "README_10Prove_orbital_localization.txt")

    with open(readme_path, "w") as f:
        f.write("10Prove_orbital_localization\n")
        f.write("=" * 60 + "\n\n")
        f.write("This folder contains quantitative proof of VBM/CBM orbital localization.\n\n")
        f.write("Main metrics:\n")
        f.write("1. IPR: larger value means stronger localization.\n")
        f.write("2. PR_fraction: smaller value means fewer effective voxels occupied.\n")
        f.write("3. Top-density weight: larger value means density concentrated in a small region.\n")
        f.write("4. Dopant-neighborhood weight: larger value means band-edge state localized near dopant.\n")
        f.write("5. VBM-CBM overlap: larger value means stronger spatial co-localization.\n")
        f.write("6. Spectral-function-style heatmap: brighter color means stronger localization.\n\n")
        f.write("Important interpretation:\n")
        f.write("Individual peak labels were removed to avoid crowding.\n")
        f.write("All cases are interpreted through the heatmap in panel (b).\n\n")
        f.write("Input root folder:\n")
        f.write(ROOT_DIR + "\n\n")
        f.write("Output files:\n")
        f.write("folder_file_inventory.csv\n")
        f.write("folder_summary.csv\n")
        f.write("orbital_localization_metrics.csv\n")
        f.write("vbm_cbm_overlap_metrics.csv\n")
        f.write("publication_summary_localization_proof.csv\n")
        f.write("Fig_IPR_localization_proof.png\n")
        f.write("Fig_PR_fraction_localization_proof.png\n")
        f.write("Fig_top_density_weight_proof.png\n")
        f.write("Fig_dopant_weight_proof.png\n")
        f.write("Fig_VBM_CBM_overlap_proof.png\n")
        f.write("Fig_spectral_function_style_localization_no_labels.png\n")
        f.write("Fig_spectral_function_style_localization_no_labels.pdf\n")

    print("\nDONE.")
    print("All results are in:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
