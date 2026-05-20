#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
"""
python3 8Convert_CubeReal_to_CubeWFK.py
"""

TEMPLATE_DIR = Path("/home/yanan/Material/Y_dope_project/CBM_VBM_Sc/Cube")
REAL_ROOT = Path("/home/yanan/Material/Y_dope_project/CBM_VBM_Sc/CubeReal")
OUT_ROOT = Path("/home/yanan/Material/Y_dope_project/CBM_VBM_Sc/8Cube_WFK_All")

DOPANTS = ["Sc", "Y", "Zr"]
CONCS = [
    ("3p125", "3.125"),
    ("6p25", "6.25"),
    ("9p375", "9.375"),
]

TASKS = [
    # dopant, conc_tag, conc_folder, state, edge, k, band

    # ---------------- Y ----------------
    ("Y", "3p125", "3.125", "MinusP", "VBM", 1, 384),
    ("Y", "3p125", "3.125", "MinusP", "CBM", 7, 385),
    ("Y", "3p125", "3.125", "PlusP",  "VBM", 1, 384),
    ("Y", "3p125", "3.125", "PlusP",  "CBM", 7, 385),

    ("Y", "6p25", "6.25", "MinusP", "VBM", 1, 383),
    ("Y", "6p25", "6.25", "MinusP", "CBM", 7, 384),
    ("Y", "6p25", "6.25", "PlusP",  "VBM", 1, 383),
    ("Y", "6p25", "6.25", "PlusP",  "CBM", 8, 384),

    ("Y", "9p375", "9.375", "MinusP", "VBM", 1, 383),
    ("Y", "9p375", "9.375", "MinusP", "CBM", 8, 384),
    ("Y", "9p375", "9.375", "PlusP",  "VBM", 1, 383),
    ("Y", "9p375", "9.375", "PlusP",  "CBM", 8, 384),

    # ---------------- Sc ----------------
    ("Sc", "3p125", "3.125", "MinusP", "VBM", 8, 384),
    ("Sc", "3p125", "3.125", "MinusP", "CBM", 5, 385),
    ("Sc", "3p125", "3.125", "PlusP",  "VBM", 1, 384),
    ("Sc", "3p125", "3.125", "PlusP",  "CBM", 3, 385),

    ("Sc", "6p25", "6.25", "MinusP", "VBM", 7, 383),
    ("Sc", "6p25", "6.25", "MinusP", "CBM", 5, 384),
    ("Sc", "6p25", "6.25", "PlusP",  "VBM", 1, 383),
    ("Sc", "6p25", "6.25", "PlusP",  "CBM", 3, 384),

    ("Sc", "9p375", "9.375", "MinusP", "VBM", 4, 383),
    ("Sc", "9p375", "9.375", "MinusP", "CBM", 5, 384),
    ("Sc", "9p375", "9.375", "PlusP",  "VBM", 1, 383),
    ("Sc", "9p375", "9.375", "PlusP",  "CBM", 6, 384),

    # ---------------- Zr ----------------
    ("Zr", "3p125", "3.125", "MinusP", "VBM", 7, 384),
    ("Zr", "3p125", "3.125", "MinusP", "CBM", 6, 385),
    ("Zr", "3p125", "3.125", "PlusP",  "VBM", 1, 384),
    ("Zr", "3p125", "3.125", "PlusP",  "CBM", 3, 385),

    ("Zr", "6p25", "6.25", "MinusP", "VBM", 7, 384),
    ("Zr", "6p25", "6.25", "MinusP", "CBM", 6, 385),
    ("Zr", "6p25", "6.25", "PlusP",  "VBM", 1, 384),
    ("Zr", "6p25", "6.25", "PlusP",  "CBM", 3, 385),

    ("Zr", "9p375", "9.375", "MinusP", "VBM", 7, 384),
    ("Zr", "9p375", "9.375", "MinusP", "CBM", 6, 385),
    ("Zr", "9p375", "9.375", "PlusP",  "VBM", 1, 384),
    ("Zr", "9p375", "9.375", "PlusP",  "CBM", 6, 385),
]


def read_cube_header(cube_path: Path):
    with cube_path.open("r") as f:
        lines = f.readlines()

    if len(lines) < 6:
        raise ValueError(f"{cube_path} is too short to be a valid cube file.")

    natoms = abs(int(float(lines[2].split()[0])))

    grid = []
    for i in range(3, 6):
        parts = lines[i].split()
        grid.append(int(float(parts[0])))

    header_len = 6 + natoms
    header = lines[:header_len]

    nx, ny, nz = grid
    npts = nx * ny * nz

    return header, grid, npts


def read_real_values(real_path: Path):
    values = []
    with real_path.open("r") as f:
        for line in f:
            for x in line.split():
                values.append(float(x))
    return values


def write_cube(out_path: Path, header, values):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for line in header:
            f.write(line)

        for i, val in enumerate(values, start=1):
            f.write(f"{val:13.5E}")
            if i % 6 == 0:
                f.write("\n")

        if len(values) % 6 != 0:
            f.write("\n")


def make_names(dopant, conc_tag, conc_folder, state, edge, k, band):
    base = f"{state}_{conc_tag}{dopant}_{edge}_k{k}_b{band}"

    template_cube = TEMPLATE_DIR / f"{base}.cube"

    real_dir = REAL_ROOT / f"{conc_folder}{dopant}_Cube_Real"
    real_file = real_dir / f"{base}_Realdata.cube_k{k}_b{band}"

    out_dir = OUT_ROOT / f"{conc_folder}{dopant}_WFK_Cube"
    out_cube = out_dir / f"{base}_wfk.cube"

    return template_cube, real_file, out_cube


def convert_one(task):
    dopant, conc_tag, conc_folder, state, edge, k, band = task

    template_cube, real_file, out_cube = make_names(
        dopant, conc_tag, conc_folder, state, edge, k, band
    )

    print("=" * 90)
    print(f"{state} {conc_folder}% {dopant}-HfO2 {edge} k{k} b{band}")
    print(f"Template : {template_cube}")
    print(f"Realdata : {real_file}")
    print(f"Output   : {out_cube}")

    if not template_cube.exists():
        print(f"[SKIP] Missing template cube: {template_cube}")
        return False

    if not real_file.exists():
        print(f"[SKIP] Missing realdata file: {real_file}")
        return False

    header, grid, npts_expected = read_cube_header(template_cube)
    values = read_real_values(real_file)

    if len(values) != npts_expected:
        raise ValueError(
            f"Grid mismatch for {real_file}\n"
            f"Template grid = {grid}, expected {npts_expected} values\n"
            f"Found {len(values)} values"
        )

    write_cube(out_cube, header, values)

    print(f"[OK] Wrote {out_cube}")
    print(f"Grid    : {grid}")
    print(f"Values  : {len(values)}")
    print(f"Min/Max : {min(values):.6e} / {max(values):.6e}")

    if min(values) < 0 < max(values):
        print("Sign    : contains both positive and negative wavefunction values")
    else:
        print("Sign    : does not contain both signs")

    return True


def main():
    print("Batch converting real-data cube files to WFK-style cube files")
    print(f"Template cube folder : {TEMPLATE_DIR}")
    print(f"Realdata root        : {REAL_ROOT}")
    print(f"Output root          : {OUT_ROOT}")
    print()

    success = 0
    skipped = 0

    for task in TASKS:
        ok = convert_one(task)
        if ok:
            success += 1
        else:
            skipped += 1

    print("=" * 90)
    print("DONE")
    print(f"Converted : {success}")
    print(f"Skipped   : {skipped}")
    print(f"Output folder: {OUT_ROOT}")


if __name__ == "__main__":
    main()
