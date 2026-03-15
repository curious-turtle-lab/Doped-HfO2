#!/usr/bin/env python3
"""
2Heat_9.375Sc.py
##########################################################################################
python3 2Heat_9.375Sc.py   --plus-den  "/home/yanan/Material/Y_dope_project/9.375Sc-HfO2-2x2x2/berry-scf-Results/berry_PlusP_9.375Sc_HfO2_12144507/berry_PlusP_9o_DEN.nc"    --minus-den  "/home/yanan/Material/Y_dope_project/9.375Sc-HfO2-2x2x2/berry-scf-Results/berry_MinusP_9.375Sc_HfO2_12144529/berry_MinusP_9o_DEN.nc"   --plus-in "/home/yanan/Material/Y_dope_project/9.375Sc-HfO2-2x2x2/berry-scf-Results/berry_PlusP_9.375Sc_HfO2_12144507/berry_PlusP_9.375Per_Sc_HfO2_2x2x2.in"  --minus-in "/home/yanan/Material/Y_dope_project/9.375Sc-HfO2-2x2x2/berry-scf-Results/berry_MinusP_9.375Sc_HfO2_12144529/berry_MinusP_9.375Per_Sc_HfO2_2x2x2.in"  --units ang --outdir 2_DEN_Figures_SHARP_9.375Sc --write-cube


############################################################################################
2D heatmaps (rho(+P), rho(-P), delta_rho) from ABINIT DEN.nc (+P/-P) and optional VESTA cube export.

Key features:
- Adds "9.375% Sc-doping" to titles
- Ensures output folder includes "9.375Sc"
- Uses interpolation='nearest' to avoid plotting blur
- Δρ ALWAYS uses symmetric colorbar: vmin=-maxabs, vmax=+maxabs
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

BOHR_TO_ANG = 0.52917721092
TAG_OUTDIR = "9.375Sc"
TITLE_PREFIX = "9.375% Sc-doping: "
# ----------------------------- Parsing .in (atoms) -----------------------------

@dataclass
class Structure:
    atomic_numbers: np.ndarray
    frac_coords: np.ndarray
    lattice_bohr: Optional[np.ndarray]


def _read_block_after_keyword(lines: List[str], key: str, nrows: int, ncols: int) -> np.ndarray:
    for i, ln in enumerate(lines):
        if ln.strip().lower() == key.lower():
            data = []
            for j in range(nrows):
                row = lines[i + 1 + j].split()
                if len(row) < ncols:
                    raise ValueError(f"Not enough columns for block '{key}'")
                data.append([float(x) for x in row[:ncols]])
            return np.array(data, dtype=float)
    raise ValueError(f"Keyword block '{key}' not found in input.")


def parse_abinit_in_structure(path_in: str) -> Structure:
    with open(path_in, "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = f.readlines()

    # strip comments and blank lines
    lines = []
    for ln in raw_lines:
        ln2 = ln.split("#", 1)[0].strip()
        if ln2:
            lines.append(ln2)

    def find_scalar_int(key: str) -> int:
        for ln in lines:
            parts = ln.split()
            if len(parts) >= 2 and parts[0].lower() == key.lower():
                return int(parts[1])
        raise ValueError(f"Missing '{key}' in {path_in}")

    def find_vector_ints(key: str) -> List[int]:
        for ln in lines:
            parts = ln.split()
            if len(parts) >= 2 and parts[0].lower() == key.lower():
                return [int(x) for x in parts[1:]]
        raise ValueError(f"Missing '{key}' in {path_in}")

    natom = find_scalar_int("natom")
    znucl = np.array(find_vector_ints("znucl"), dtype=int)

    # typat spans multiple lines
    typat_vals: List[int] = []
    typat_started = False
    for ln in lines:
        parts = ln.split()
        if parts and parts[0].lower() == "typat":
            typat_started = True
            typat_vals.extend([int(x) for x in parts[1:]])
            continue
        if typat_started:
            # stop if we hit another keyword (heuristic)
            if parts and parts[0].isalpha() and parts[0].lower() not in ("typat",):
                break
            typat_vals.extend([int(x) for x in parts])
            if len(typat_vals) >= natom:
                break
    if len(typat_vals) < natom:
        raise ValueError(f"Could not read {natom} typat values from {path_in}")
    typat = np.array(typat_vals[:natom], dtype=int)

    # xred block
    xred = None
    for i, ln in enumerate(lines):
        if ln.strip().lower() == "xred":
            data = []
            for j in range(natom):
                row = lines[i + 1 + j].split()
                if len(row) < 3:
                    raise ValueError(f"Bad xred row in {path_in}")
                data.append([float(row[0]), float(row[1]), float(row[2])])
            xred = np.array(data, dtype=float)
            break
    if xred is None:
        raise ValueError(f"Missing xred block in {path_in}")

    # optional lattice from acell+rprim
    lattice_bohr = None
    acell = None
    for ln in lines:
        parts = ln.split()
        if parts and parts[0].lower() == "acell":
            if len(parts) < 4:
                raise ValueError(f"acell line malformed in {path_in}")
            acell = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
            break
    if acell is not None:
        try:
            rprim = _read_block_after_keyword(lines, "rprim", 3, 3)
            lattice_bohr = (acell.reshape(3, 1) * rprim).astype(float)
        except Exception:
            lattice_bohr = None

    atomic_numbers = znucl[typat - 1]
    return Structure(atomic_numbers=atomic_numbers, frac_coords=xred, lattice_bohr=lattice_bohr)


# ----------------------------- Reading DEN.nc density -----------------------------

@dataclass
class DensityGrid:
    rho: np.ndarray
    lattice_bohr: np.ndarray
    ng: Tuple[int, int, int]


def read_abinit_den_nc(path_den: str) -> DensityGrid:
    ds = nc.Dataset(path_den)

    if "density" not in ds.variables:
        raise ValueError(f"'density' not in {path_den}. Vars: {list(ds.variables.keys())[:30]}")

    den = np.array(ds.variables["density"][:], dtype=float)
    den = np.squeeze(den)
    if den.ndim != 3:
        raise ValueError(f"After squeeze expected 3D grid, got shape {den.shape} from {path_den}")

    if "primitive_vectors" not in ds.variables:
        raise ValueError(f"'primitive_vectors' not found in {path_den}")
    lattice_bohr = np.array(ds.variables["primitive_vectors"][:], dtype=float)

    ds.close()
    return DensityGrid(rho=den, lattice_bohr=lattice_bohr, ng=tuple(den.shape))


# ----------------------------- 2D plane extraction -----------------------------

def _axis_map():
    # rho[i,j,k] -> x,y,z
    return {"x": 0, "y": 1, "z": 2}


def plane_average_to_2d(rho3d: np.ndarray, plane: str = "xz", avg_axis: str = "y") -> np.ndarray:
    plane = plane.lower()
    avg_axis = avg_axis.lower()
    amap = _axis_map()

    if plane == "xz":
        if avg_axis != "y":
            raise ValueError("For plane xz, avg_axis must be y")
        return rho3d.mean(axis=amap["y"])  # (nx, nz)

    if plane == "xy":
        if avg_axis != "z":
            raise ValueError("For plane xy, avg_axis must be z")
        return rho3d.mean(axis=amap["z"])  # (nx, ny)

    if plane == "yz":
        if avg_axis != "x":
            raise ValueError("For plane yz, avg_axis must be x")
        return rho3d.mean(axis=amap["x"])  # (ny, nz)

    raise ValueError("plane must be xz, xy, yz")


def make_extent_ang(lattice_bohr: np.ndarray, ng: Tuple[int, int, int], plane: str) -> List[float]:
    """extent in Angstrom for imshow: [x0, x1, z0, z1] etc."""
    plane = plane.lower()
    lat_ang = lattice_bohr * BOHR_TO_ANG

    a1 = np.linalg.norm(lat_ang[0])
    a2 = np.linalg.norm(lat_ang[1])
    a3 = np.linalg.norm(lat_ang[2])

    nx, ny, nz = ng

    if plane == "xz":
        ax1 = np.linspace(0.0, a1, nx, endpoint=False)
        ax2 = np.linspace(0.0, a3, nz, endpoint=False)
    elif plane == "xy":
        ax1 = np.linspace(0.0, a1, nx, endpoint=False)
        ax2 = np.linspace(0.0, a2, ny, endpoint=False)
    elif plane == "yz":
        ax1 = np.linspace(0.0, a2, ny, endpoint=False)
        ax2 = np.linspace(0.0, a3, nz, endpoint=False)
    else:
        raise ValueError("plane must be xz, xy, yz")

    dx = ax1[1] - ax1[0] if len(ax1) > 1 else 1.0
    dy = ax2[1] - ax2[0] if len(ax2) > 1 else 1.0

    return [ax1[0], ax1[-1] + dx, ax2[0], ax2[-1] + dy]


# ----------------------------- Cube export for VESTA -----------------------------

def write_gaussian_cube(path_cube: str,
                        lattice_bohr: np.ndarray,
                        atomic_numbers: np.ndarray,
                        frac_coords: np.ndarray,
                        rho3d: np.ndarray,
                        comment1: str,
                        comment2: str) -> None:
    natom = len(atomic_numbers)
    nx, ny, nz = rho3d.shape

    cart_bohr = frac_coords @ lattice_bohr

    a1 = lattice_bohr[0] / nx
    a2 = lattice_bohr[1] / ny
    a3 = lattice_bohr[2] / nz

    with open(path_cube, "w", encoding="utf-8") as f:
        f.write(f"{comment1}\n{comment2}\n")
        f.write(f"{natom:5d} {0.0:12.6f} {0.0:12.6f} {0.0:12.6f}\n")
        f.write(f"{nx:5d} {a1[0]:12.6f} {a1[1]:12.6f} {a1[2]:12.6f}\n")
        f.write(f"{ny:5d} {a2[0]:12.6f} {a2[1]:12.6f} {a2[2]:12.6f}\n")
        f.write(f"{nz:5d} {a3[0]:12.6f} {a3[1]:12.6f} {a3[2]:12.6f}\n")

        for Z, r in zip(atomic_numbers, cart_bohr):
            f.write(f"{int(Z):5d} {float(Z):12.6f} {r[0]:12.6f} {r[1]:12.6f} {r[2]:12.6f}\n")

        count = 0
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{rho3d[i, j, k]:13.5e} ")
                    count += 1
                    if count % 6 == 0:
                        f.write("\n")
                if count % 6 != 0:
                    f.write("\n")


# ----------------------------- Plotting -----------------------------

def plot_heatmap(path_base: str,
                 data2d: np.ndarray,
                 extent: List[float],
                 xlabel: str, ylabel: str, title: str,
                 units_label: str,
                 dpi: int = 800,
                 interpolation: str = "nearest",
                 square_pixels: bool = True,
                 vmin: Optional[float] = None,
                 vmax: Optional[float] = None,
                 cmap: Optional[str] = None) -> None:
    img = data2d.T  # imshow wants [row(y), col(x)] => transpose for our extent convention

    plt.figure()
    aspect = "equal" if square_pixels else "auto"
    plt.imshow(
        img,
        origin="lower",
        extent=extent,
        aspect=aspect,
        interpolation=interpolation,  # no blur
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    cb = plt.colorbar()
    cb.set_label(units_label)
    plt.tight_layout()
    plt.savefig(path_base + ".png", dpi=dpi)
    plt.close()


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="2D heatmaps & VESTA cube export from ABINIT DEN.nc (+P/-P).")

    ap.add_argument("--plus-den", required=True, help="Path to +P DEN.nc")
    ap.add_argument("--minus-den", required=True, help="Path to -P DEN.nc")
    ap.add_argument("--plus-in", required=True, help="Path to +P .in (for atoms)")
    ap.add_argument("--minus-in", required=True, help="Path to -P .in (for atoms)")

    ap.add_argument("--outdir", default="2_DEN_Figures_SHARP_9.375Sc", help="Base output directory name")
    ap.add_argument("--plane", default="xz", choices=["xz", "xy", "yz"])
    ap.add_argument("--avg-axis", default="y", choices=["x", "y", "z"])
    ap.add_argument("--units", default="ang", choices=["bohr", "ang"])

    ap.add_argument("--write-cube", action="store_true")
    ap.add_argument("--interpolation", default="nearest",
                    choices=["nearest", "none", "bilinear", "bicubic"])
    ap.add_argument("--dpi", type=int, default=800)
    ap.add_argument("--square-pixels", action="store_true")

    args = ap.parse_args()

    # Ensure output folder includes 9.375Sc
    outdir = args.outdir.rstrip("/").rstrip("_")
    if TAG_OUTDIR.lower() not in outdir.lower():
        outdir = f"{outdir}_{TAG_OUTDIR}"
    os.makedirs(outdir, exist_ok=True)

    plus = read_abinit_den_nc(args.plus_den)
    minus = read_abinit_den_nc(args.minus_den)
    if plus.ng != minus.ng:
        raise ValueError(f"Grid mismatch: plus.ng={plus.ng} vs minus.ng={minus.ng}")

    rho_plus = plus.rho
    rho_minus = minus.rho
    drho = rho_plus - rho_minus

    # units conversion
    if args.units == "ang":
        conv = 1.0 / (BOHR_TO_ANG ** 3)  # e/bohr^3 -> e/Å^3
        rho_plus_u = rho_plus * conv
        rho_minus_u = rho_minus * conv
        drho_u = drho * conv
        unit_label = "e/Å$^3$"
        xlabel_unit = "Å"
    else:
        rho_plus_u = rho_plus
        rho_minus_u = rho_minus
        drho_u = drho
        unit_label = "e/bohr$^3$"
        xlabel_unit = "bohr"

    extent = make_extent_ang(plus.lattice_bohr, plus.ng, args.plane)

    if args.plane == "xz":
        xlabel, ylabel = f"x ({xlabel_unit})", f"z ({xlabel_unit})"
    elif args.plane == "xy":
        xlabel, ylabel = f"x ({xlabel_unit})", f"y ({xlabel_unit})"
    else:
        xlabel, ylabel = f"y ({xlabel_unit})", f"z ({xlabel_unit})"

    plus2 = plane_average_to_2d(rho_plus_u, plane=args.plane, avg_axis=args.avg_axis)
    minus2 = plane_average_to_2d(rho_minus_u, plane=args.plane, avg_axis=args.avg_axis)
    delta2 = plane_average_to_2d(drho_u, plane=args.plane, avg_axis=args.avg_axis)

    # Δρ symmetric color scale ALWAYS
    dv = float(np.max(np.abs(delta2)))
    if not np.isfinite(dv) or dv == 0.0:
        dv = 1.0

    plot_heatmap(
        os.path.join(outdir, f"rho_plus_{args.plane}_av{args.avg_axis}_{args.units}_{TAG_OUTDIR}"),
        plus2, extent, xlabel, ylabel,
        title=TITLE_PREFIX + f"$\\rho(+P)$ {args.plane}-plane (avg {args.avg_axis})",
        units_label=unit_label,
        dpi=args.dpi,
        interpolation=args.interpolation,
        square_pixels=args.square_pixels,
        cmap=None,
    )

    plot_heatmap(
        os.path.join(outdir, f"rho_minus_{args.plane}_av{args.avg_axis}_{args.units}_{TAG_OUTDIR}"),
        minus2, extent, xlabel, ylabel,
        title=TITLE_PREFIX + f"$\\rho(-P)$ {args.plane}-plane (avg {args.avg_axis})",
        units_label=unit_label,
        dpi=args.dpi,
        interpolation=args.interpolation,
        square_pixels=args.square_pixels,
        cmap=None,
    )

    # Δρ: symmetric + diverging colormap (best for +/-)
    plot_heatmap(
        os.path.join(outdir, f"delta_rho_{args.plane}_av{args.avg_axis}_{args.units}_{TAG_OUTDIR}"),
        delta2, extent, xlabel, ylabel,
        title=TITLE_PREFIX + f"$\\Delta\\rho=\\rho(+P)-\\rho(-P)$ {args.plane}-plane (avg {args.avg_axis})",
        units_label=unit_label,
        dpi=args.dpi,
        interpolation=args.interpolation,
        square_pixels=args.square_pixels,
        vmin=-dv, vmax=dv,
        cmap="RdBu_r",
    )

    if args.write_cube:
        st_plus = parse_abinit_in_structure(args.plus_in)
        st_minus = parse_abinit_in_structure(args.minus_in)

        cube_plus = os.path.join(outdir, f"rho_plus_{TAG_OUTDIR}.cube")
        cube_minus = os.path.join(outdir, f"rho_minus_{TAG_OUTDIR}.cube")
        cube_delta = os.path.join(outdir, f"delta_rho_plus_minus_{TAG_OUTDIR}.cube")

        write_gaussian_cube(
            cube_plus, plus.lattice_bohr, st_plus.atomic_numbers, st_plus.frac_coords, rho_plus_u,
            comment1="rho(+P) from ABINIT DEN.nc",
            comment2=f"Units: {unit_label}",
        )
        write_gaussian_cube(
            cube_minus, plus.lattice_bohr, st_minus.atomic_numbers, st_minus.frac_coords, rho_minus_u,
            comment1="rho(-P) from ABINIT DEN.nc",
            comment2=f"Units: {unit_label}",
        )
        write_gaussian_cube(
            cube_delta, plus.lattice_bohr, st_plus.atomic_numbers, st_plus.frac_coords, drho_u,
            comment1="Delta rho = rho(+P) - rho(-P)",
            comment2=f"Units: {unit_label}",
        )
        print("[OK] cube files written.")

    print(f"[OK] Output folder: {outdir}")


if __name__ == "__main__":
    main()

