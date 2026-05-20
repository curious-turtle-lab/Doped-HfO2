#!/usr/bin/env python3
"""
Advanced CBM cube comparison for ferroelectric ±P states.

This version supports:
1. python3 3cbm_difference_advanced.py \
    --batch-all \
    --base-dir ./Cube \
    --radius 1.8 \
    --save-npy

2. Batch-all mode for:
      Y, Sc, Zr at 3.125%, 6.25%, 9.375%
   using CBM k-point / band indices from your summary.


python3 3cbm_difference_advanced.py \
    --batch-all \
    --base-dir ./Cube \
    --radius 1.8 \
    --save-npy

Notes
-----
- Radius is in the same length unit as the cube coordinates.
- For most cube files exported from electronic-structure codes, this is often Bohr.
- If your cube coordinates are in Angstrom, use Angstrom consistently.
- This script compares CBM cube files for PlusP vs MinusP.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Hardcoded batch systems from your VBM summary
# Based on the uploaded VBM/CBM results summary.
# Update cube filenames here if any local filename differs.
# ============================================================
SYSTEMS = [
    # ===================== 3.125% =====================
    # ----- Sc -----
    {
        "dopant": "Sc", "conc": "3.125%", "tag": "3p125Sc",
        "plus_k": 3, "plus_b": 385,
        "minus_k": 5, "minus_b": 385,
    },
    # ----- Y -----
    {
        "dopant": "Y", "conc": "3.125%", "tag": "3p125Y",
        "plus_k": 7, "plus_b": 385,
        "minus_k": 7, "minus_b": 385,
    },
    # ----- Zr -----
    {
        "dopant": "Zr", "conc": "3.125%", "tag": "3p125Zr",
        "plus_k": 3, "plus_b": 385,
        "minus_k": 6, "minus_b": 385,
    },

    # ===================== 6.25% =====================
    # ----- Sc -----
    {
        "dopant": "Sc", "conc": "6.25%", "tag": "6p25Sc",
        "plus_k": 3, "plus_b": 384,
        "minus_k": 5, "minus_b": 384,
    },
    # ----- Y -----
    {
        "dopant": "Y", "conc": "6.25%", "tag": "6p25Y",
        "plus_k": 8, "plus_b": 384,
        "minus_k": 7, "minus_b": 384,
    },
    # ----- Zr -----
    {
        "dopant": "Zr", "conc": "6.25%", "tag": "6p25Zr",
        "plus_k": 3, "plus_b": 385,
        "minus_k": 6, "minus_b": 385,
    },

    # ===================== 9.375% =====================
    # ----- Sc -----
    {
        "dopant": "Sc", "conc": "9.375%", "tag": "9p375Sc",
        "plus_k": 6, "plus_b": 384,
        "minus_k": 5, "minus_b": 384,
    },
    # ----- Y -----
    {
        "dopant": "Y", "conc": "9.375%", "tag": "9p375Y",
        "plus_k": 8, "plus_b": 384,
        "minus_k": 8, "minus_b": 384,
    },
    # ----- Zr -----
    {
        "dopant": "Zr", "conc": "9.375%", "tag": "9p375Zr",
        "plus_k": 6, "plus_b": 385,
        "minus_k": 6, "minus_b": 385,
    },
]


# ============================================================
# Periodic table mapping used in cube atom parsing
# ============================================================
Z_TO_SYMBOL = {
    1: "H",
    8: "O",
    21: "Sc",
    39: "Y",
    40: "Zr",
    72: "Hf",
}


@dataclass
class Atom:
    Z: int
    charge: float
    x: float
    y: float
    z: float

    @property
    def symbol(self) -> str:
        return Z_TO_SYMBOL.get(self.Z, f"Z{self.Z}")

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass
class CubeData:
    title1: str
    title2: str
    natoms: int
    origin: np.ndarray
    dims: Tuple[int, int, int]
    axes: np.ndarray
    atoms: List[Atom]
    values: np.ndarray


# ============================================================
# IO
# ============================================================
def read_cube(filename: str) -> CubeData:
    with open(filename, "r") as f:
        lines = f.readlines()

    title1 = lines[0].rstrip("\n")
    title2 = lines[1].rstrip("\n")

    parts = lines[2].split()
    natoms = int(parts[0])
    origin = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)

    grid_info = []
    for i in range(3):
        parts = lines[3 + i].split()
        n = int(parts[0])
        vec = [float(parts[1]), float(parts[2]), float(parts[3])]
        grid_info.append((n, vec))

    nx, ax = grid_info[0]
    ny, ay = grid_info[1]
    nz, az = grid_info[2]
    axes = np.array([ax, ay, az], dtype=float)

    atom_start = 6
    atom_end = atom_start + abs(natoms)
    atoms: List[Atom] = []
    for i in range(atom_start, atom_end):
        parts = lines[i].split()
        Z = int(float(parts[0]))
        charge = float(parts[1])
        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
        atoms.append(Atom(Z, charge, x, y, z))

    data_tokens = []
    for line in lines[atom_end:]:
        data_tokens.extend(line.split())

    data = np.array([float(x) for x in data_tokens], dtype=float)
    expected = nx * ny * nz
    if data.size != expected:
        raise ValueError(f"{filename}: expected {expected} values, got {data.size}")

    values = data.reshape((nx, ny, nz), order="C")

    return CubeData(
        title1=title1,
        title2=title2,
        natoms=natoms,
        origin=origin,
        dims=(nx, ny, nz),
        axes=axes,
        atoms=atoms,
        values=values,
    )


# ============================================================
# Math / geometry helpers
# ============================================================
def voxel_volume(cube: CubeData) -> float:
    return abs(np.linalg.det(cube.axes))


def integrated(arr: np.ndarray, dv: float) -> float:
    return float(np.sum(arr) * dv)


def grid_point_coordinates(cube: CubeData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = cube.dims

    i = np.arange(nx)[:, None, None, None]
    j = np.arange(ny)[None, :, None, None]
    k = np.arange(nz)[None, None, :, None]

    origin = cube.origin[None, None, None, :]
    a1 = cube.axes[0][None, None, None, :]
    a2 = cube.axes[1][None, None, None, :]
    a3 = cube.axes[2][None, None, None, :]

    r = origin + i * a1 + j * a2 + k * a3

    X = r[..., 0]
    Y = r[..., 1]
    Z = r[..., 2]
    return X, Y, Z


def central_slice(arr: np.ndarray, axis: str) -> np.ndarray:
    nx, ny, nz = arr.shape
    if axis == "x":
        return arr[nx // 2, :, :]
    if axis == "y":
        return arr[:, ny // 2, :]
    if axis == "z":
        return arr[:, :, nz // 2]
    raise ValueError("axis must be x, y, or z")


def plane_average_z(arr: np.ndarray) -> np.ndarray:
    return np.mean(arr, axis=(0, 1))


def build_species_masks(
    cube: CubeData,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    radius: float
) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {}
    species_list = sorted(set(atom.symbol for atom in cube.atoms))

    for sp in species_list:
        mask = np.zeros(cube.dims, dtype=bool)
        atoms_sp = [a for a in cube.atoms if a.symbol == sp]
        for atom in atoms_sp:
            dx = X - atom.x
            dy = Y - atom.y
            dz = Z - atom.z
            dist2 = dx * dx + dy * dy + dz * dz
            mask |= (dist2 <= radius * radius)
        masks[sp] = mask

    return masks


def species_projection(arr: np.ndarray, masks: Dict[str, np.ndarray], dv: float) -> Dict[str, float]:
    proj = {}
    for sp, mask in masks.items():
        proj[sp] = float(np.sum(arr[mask]) * dv)
    return proj


def z_center_of_density(arr: np.ndarray, Z: np.ndarray, dv: float) -> float:
    denom = np.sum(arr) * dv
    if abs(denom) < 1e-30:
        return np.nan
    numer = np.sum(arr * Z) * dv
    return float(numer / denom)


def z_center_abs(arr: np.ndarray, Z: np.ndarray, dv: float) -> float:
    w = np.abs(arr)
    denom = np.sum(w) * dv
    if denom < 1e-30:
        return np.nan
    numer = np.sum(w * Z) * dv
    return float(numer / denom)


# ============================================================
# Plot helpers
# ============================================================
def save_three_panel_density(
    arr1: np.ndarray,
    arr2: np.ndarray,
    arr3: np.ndarray,
    title1: str,
    title2: str,
    title3: str,
    outfile: str,
    cmap_density: str = "viridis",
    cmap_diff: str = "RdBu_r"
) -> None:
    vmax3 = np.max(np.abs(arr3))
    if vmax3 == 0:
        vmax3 = 1e-12

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    im1 = axes[0].imshow(arr1.T, origin="lower", cmap=cmap_density, aspect="auto")
    axes[0].set_title(title1)
    axes[0].set_xlabel("x index")
    axes[0].set_ylabel("y index")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(arr2.T, origin="lower", cmap=cmap_density, aspect="auto")
    axes[1].set_title(title2)
    axes[1].set_xlabel("x index")
    axes[1].set_ylabel("y index")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    im3 = axes[2].imshow(arr3.T, origin="lower", cmap=cmap_diff, vmin=-vmax3, vmax=vmax3, aspect="auto")
    axes[2].set_title(title3)
    axes[2].set_xlabel("x index")
    axes[2].set_ylabel("y index")
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close(fig)


def save_three_panel_signed(
    arr1: np.ndarray,
    arr2: np.ndarray,
    arr3: np.ndarray,
    title1: str,
    title2: str,
    title3: str,
    outfile: str
) -> None:
    vmax = max(np.max(np.abs(arr1)), np.max(np.abs(arr2)), np.max(np.abs(arr3)))
    if vmax == 0:
        vmax = 1e-12

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, arr, title in zip(axes, [arr1, arr2, arr3], [title1, title2, title3]):
        im = ax.imshow(arr.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close(fig)


def save_z_profile(
    zvals: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    pdiff: np.ndarray,
    label1: str,
    label2: str,
    outfile: str
) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(zvals, p1, label=label1)
    plt.plot(zvals, p2, label=label2)
    plt.plot(zvals, pdiff, label="Difference")
    plt.xlabel("z coordinate")
    plt.ylabel("Plane-averaged value")
    plt.title("Plane-averaged profile along z")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


# ============================================================
# Core analysis
# ============================================================
def analyze_pair(
    cube1_path: str,
    cube2_path: str,
    label1: str,
    label2: str,
    outdir: str,
    radius: float = 1.8,
    save_npy: bool = False
) -> None:
    os.makedirs(outdir, exist_ok=True)

    cube1 = read_cube(cube1_path)
    cube2 = read_cube(cube2_path)

    if cube1.dims != cube2.dims:
        raise ValueError(f"Grid mismatch: {cube1.dims} vs {cube2.dims}")
    if not np.allclose(cube1.origin, cube2.origin, atol=1e-8):
        raise ValueError("Origins differ")
    if not np.allclose(cube1.axes, cube2.axes, atol=1e-8):
        raise ValueError("Voxel vectors differ")

    psi1 = cube1.values
    psi2 = cube2.values
    rho1 = psi1 ** 2
    rho2 = psi2 ** 2
    delta_rho = rho1 - rho2
    delta_psi = psi1 - psi2

    dv = voxel_volume(cube1)
    X, Y, Z = grid_point_coordinates(cube1)

    masks = build_species_masks(cube1, X, Y, Z, radius=radius)
    proj_rho1 = species_projection(rho1, masks, dv)
    proj_rho2 = species_projection(rho2, masks, dv)
    proj_abs_drho = species_projection(np.abs(delta_rho), masks, dv)
    proj_abs_dpsi = species_projection(np.abs(delta_psi), masks, dv)

    zc_rho1 = z_center_of_density(rho1, Z, dv)
    zc_rho2 = z_center_of_density(rho2, Z, dv)
    zc_abs_drho = z_center_abs(delta_rho, Z, dv)
    zc_abs_dpsi = z_center_abs(delta_psi, Z, dv)

    z_profile1 = plane_average_z(rho1)
    z_profile2 = plane_average_z(rho2)
    z_profile_diff = plane_average_z(delta_rho)

    nz = cube1.dims[2]
    zvals = cube1.origin[2] + np.arange(nz) * np.linalg.norm(cube1.axes[2])

    np.savetxt(
        os.path.join(outdir, "z_profile.dat"),
        np.column_stack([zvals, z_profile1, z_profile2, z_profile_diff]),
        header=f"z {label1}_rho {label2}_rho delta_rho"
    )

    save_z_profile(
        zvals, z_profile1, z_profile2, z_profile_diff,
        label1, label2,
        os.path.join(outdir, "z_profile.png")
    )

    s1_rho = central_slice(rho1, "z")
    s2_rho = central_slice(rho2, "z")
    sd_rho = central_slice(delta_rho, "z")

    save_three_panel_density(
        s1_rho, s2_rho, sd_rho,
        f"{label1}: |ψ|²",
        f"{label2}: |ψ|²",
        "Δ|ψ|²",
        os.path.join(outdir, "density_zslice.png")
    )

    s1_psi = central_slice(psi1, "z")
    s2_psi = central_slice(psi2, "z")
    sd_psi = central_slice(delta_psi, "z")

    save_three_panel_signed(
        s1_psi, s2_psi, sd_psi,
        f"{label1}: ψ",
        f"{label2}: ψ",
        "Δψ",
        os.path.join(outdir, "signedpsi_zslice.png")
    )

    for axis in ["x", "y", "z"]:
        s1 = central_slice(rho1, axis)
        s2 = central_slice(rho2, axis)
        sd = central_slice(delta_rho, axis)
        save_three_panel_density(
            s1, s2, sd,
            f"{label1}: |ψ|² ({axis}-slice)",
            f"{label2}: |ψ|² ({axis}-slice)",
            f"Δ|ψ|² ({axis}-slice)",
            os.path.join(outdir, f"density_{axis}slice.png")
        )

        p1 = central_slice(psi1, axis)
        p2 = central_slice(psi2, axis)
        pd = central_slice(delta_psi, axis)
        save_three_panel_signed(
            p1, p2, pd,
            f"{label1}: ψ ({axis}-slice)",
            f"{label2}: ψ ({axis}-slice)",
            f"Δψ ({axis}-slice)",
            os.path.join(outdir, f"signedpsi_{axis}slice.png")
        )

    if save_npy:
        np.save(os.path.join(outdir, "psi1.npy"), psi1)
        np.save(os.path.join(outdir, "psi2.npy"), psi2)
        np.save(os.path.join(outdir, "rho1.npy"), rho1)
        np.save(os.path.join(outdir, "rho2.npy"), rho2)
        np.save(os.path.join(outdir, "delta_rho.npy"), delta_rho)
        np.save(os.path.join(outdir, "delta_psi.npy"), delta_psi)

    total_rho1 = integrated(rho1, dv)
    total_rho2 = integrated(rho2, dv)
    total_abs_drho = integrated(np.abs(delta_rho), dv)
    total_abs_dpsi = integrated(np.abs(delta_psi), dv)

    species_all = sorted(masks.keys())

    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write("Advanced CBM difference analysis\n")
        f.write("================================\n\n")
        f.write(f"cube1 = {cube1_path}\n")
        f.write(f"cube2 = {cube2_path}\n")
        f.write(f"label1 = {label1}\n")
        f.write(f"label2 = {label2}\n")
        f.write(f"radius = {radius}\n\n")

        f.write(f"dims = {cube1.dims}\n")
        f.write(f"origin = {cube1.origin}\n")
        f.write(f"axes =\n{cube1.axes}\n\n")
        f.write(f"voxel volume = {dv:.10e}\n\n")

        f.write("Global integrals\n")
        f.write("----------------\n")
        f.write(f"∫rho1 dV              = {total_rho1:.10e}\n")
        f.write(f"∫rho2 dV              = {total_rho2:.10e}\n")
        f.write(f"∫|delta_rho| dV       = {total_abs_drho:.10e}\n")
        f.write(f"∫|delta_psi| dV       = {total_abs_dpsi:.10e}\n")
        f.write(f"A_rho = ∫|Δrho|/(∫rho1+∫rho2) = {total_abs_drho / max(total_rho1 + total_rho2, 1e-30):.10e}\n\n")

        f.write("z-centers\n")
        f.write("---------\n")
        f.write(f"z_center(rho1)        = {zc_rho1:.10f}\n")
        f.write(f"z_center(rho2)        = {zc_rho2:.10f}\n")
        f.write(f"z_center(|delta_rho|) = {zc_abs_drho:.10f}\n")
        f.write(f"z_center(|delta_psi|) = {zc_abs_dpsi:.10f}\n")
        f.write(f"Δz_center(rho1-rho2)  = {zc_rho1 - zc_rho2:.10f}\n\n")

        f.write("Species-projected integrals\n")
        f.write("---------------------------\n")
        for sp in species_all:
            f.write(
                f"{sp:>4s}  "
                f"rho1={proj_rho1[sp]:.10e}  "
                f"rho2={proj_rho2[sp]:.10e}  "
                f"|Δrho|={proj_abs_drho[sp]:.10e}  "
                f"|Δpsi|={proj_abs_dpsi[sp]:.10e}\n"
            )

        f.write("\nSpecies fractions\n")
        f.write("-----------------\n")
        for sp in species_all:
            frac1 = proj_rho1[sp] / total_rho1 if total_rho1 > 0 else np.nan
            frac2 = proj_rho2[sp] / total_rho2 if total_rho2 > 0 else np.nan
            fracd = proj_abs_drho[sp] / total_abs_drho if total_abs_drho > 0 else np.nan
            f.write(
                f"{sp:>4s}  "
                f"frac_rho1={frac1:.6f}  "
                f"frac_rho2={frac2:.6f}  "
                f"frac_|Δrho|={fracd:.6f}\n"
            )

    print(f"Done. Output written to: {outdir}")
    print("Main outputs:")
    print("  summary.txt")
    print("  density_zslice.png")
    print("  signedpsi_zslice.png")
    print("  z_profile.png")


# ============================================================
# Batch helpers
# ============================================================
def default_cube_names_from_system(system):
    dopant = system["dopant"]
    tag = system["tag"]
    conc = system["conc"]

    plus_k = system["plus_k"]
    plus_b = system["plus_b"]
    minus_k = system["minus_k"]
    minus_b = system["minus_b"]

    cube1 = f"PlusP_{tag}_CBM_k{plus_k}_b{plus_b}.cube"
    cube2 = f"MinusP_{tag}_CBM_k{minus_k}_b{minus_b}.cube"

    label1 = f"PlusP_CBM_{tag}"
    label2 = f"MinusP_CBM_{tag}"

    outdir = f"3cbm_{tag}_adv_out"

    print(f"\n=== {dopant} {conc} ===")
    print(f"cube1 : {cube1}")
    print(f"cube2 : {cube2}")
    print(f"outdir: {outdir}")

    return cube1, cube2, label1, label2, outdir


def run_batch_all(
    base_dir: str,
    radius: float,
    save_npy: bool,
    master_results_dir: str
) -> None:
    missing: List[str] = []
    completed = 0

    for system in SYSTEMS:
        cube1_name, cube2_name, label1, label2, outdir_name = default_cube_names_from_system(system)

        cube1_path = os.path.join(base_dir, cube1_name)
        cube2_path = os.path.join(base_dir, cube2_name)
        outdir = os.path.join(master_results_dir, outdir_name)

        if not os.path.exists(cube1_path):
            missing.append(cube1_path)
            print(f"Missing file: {cube1_path}")
            continue

        if not os.path.exists(cube2_path):
            missing.append(cube2_path)
            print(f"Missing file: {cube2_path}")
            continue

        analyze_pair(
            cube1_path=cube1_path,
            cube2_path=cube2_path,
            label1=label1,
            label2=label2,
            outdir=outdir,
            radius=radius,
            save_npy=save_npy,
        )
        completed += 1

    print("\n========================================")
    print(f"Completed analyses: {completed}/{len(SYSTEMS)}")
    if missing:
        print("Missing cube files:")
        for m in missing:
            print(f"  {m}")
    else:
        print("All batch cube files were found.")
    print("========================================")


# ============================================================
# Main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser()

    # single-pair mode
    parser.add_argument("--cube1", help="First cube file, usually PlusP CBM cube")
    parser.add_argument("--cube2", help="Second cube file, usually MinusP CBM cube")
    parser.add_argument("--label1", default="State1")
    parser.add_argument("--label2", default="State2")
    parser.add_argument("--outdir", default="cbm_adv_out")

    # batch mode
    parser.add_argument("--batch-all", action="store_true", help="Run all Sc/Y/Zr CBM cases at 3.125, 6.25, 9.375")
    parser.add_argument("--base-dir", default=".", help="Directory containing all cube files")

    # common
    parser.add_argument("--radius", type=float, default=1.8, help="Projection radius around atoms")
    parser.add_argument("--save-npy", action="store_true")

    args = parser.parse_args()

    # ============================================================
    # Create master results folder beside this Python script
    # ============================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    master_results_dir = os.path.join(
        script_dir,
        "3cbm_difference_advanced_Results"
    )
    os.makedirs(master_results_dir, exist_ok=True)

    if args.batch_all:
        run_batch_all(
            base_dir=args.base_dir,
            radius=args.radius,
            save_npy=args.save_npy,
            master_results_dir=master_results_dir,
        )
        return

    if not args.cube1 or not args.cube2:
        raise SystemExit(
            "For single mode, provide --cube1 and --cube2.\n"
            "Or use --batch-all for all predefined systems."
        )

    # For single-pair mode, also place output inside master results folder
    single_outdir = args.outdir
    if not os.path.isabs(single_outdir):
        single_outdir = os.path.join(master_results_dir, single_outdir)

    analyze_pair(
        cube1_path=args.cube1,
        cube2_path=args.cube2,
        label1=args.label1,
        label2=args.label2,
        outdir=single_outdir,
        radius=args.radius,
        save_npy=args.save_npy,
    )


if __name__ == "__main__":
    main()
