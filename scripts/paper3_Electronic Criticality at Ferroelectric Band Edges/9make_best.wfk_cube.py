#!/usr/bin/env python3
from pathlib import Path
import numpy as np

BOHR_TO_ANG = 0.529177210903

BASE_DIR = Path("/home/yanan/Material/Y_dope_project/CBM_VBM_Sc")
INPUT_ROOT = BASE_DIR / "8Cube_WFK_All"
OUTPUT_ROOT = BASE_DIR / "9Cube_WFK_All_Best"

Z_TO_ELEM = {8: "O", 21: "Sc", 39: "Y", 40: "Zr", 72: "Hf"}
DOPANT_Z = {"Sc": 21, "Y": 39, "Zr": 40}

CROP_RADIUS = {"Sc": 3.5, "Y": 3.8, "Zr": 3.6}
O_CUTOFF = {"Sc": 2.8, "Y": 3.0, "Zr": 2.8}
HF_CUTOFF = {"Sc": 3.6, "Y": 3.8, "Zr": 3.6}

TARGET_FILES = [
    # Y
    "3.125Y_WFK_Cube/MinusP_3p125Y_VBM_k1_b384_wfk.cube",
    "3.125Y_WFK_Cube/MinusP_3p125Y_CBM_k7_b385_wfk.cube",
    "3.125Y_WFK_Cube/PlusP_3p125Y_VBM_k1_b384_wfk.cube",
    "3.125Y_WFK_Cube/PlusP_3p125Y_CBM_k7_b385_wfk.cube",

    "6.25Y_WFK_Cube/MinusP_6p25Y_VBM_k1_b383_wfk.cube",
    "6.25Y_WFK_Cube/MinusP_6p25Y_CBM_k7_b384_wfk.cube",
    "6.25Y_WFK_Cube/PlusP_6p25Y_VBM_k1_b383_wfk.cube",
    "6.25Y_WFK_Cube/PlusP_6p25Y_CBM_k8_b384_wfk.cube",

    "9.375Y_WFK_Cube/MinusP_9p375Y_VBM_k1_b383_wfk.cube",
    "9.375Y_WFK_Cube/MinusP_9p375Y_CBM_k8_b384_wfk.cube",
    "9.375Y_WFK_Cube/PlusP_9p375Y_VBM_k1_b383_wfk.cube",
    "9.375Y_WFK_Cube/PlusP_9p375Y_CBM_k8_b384_wfk.cube",

    # Sc
    "3.125Sc_WFK_Cube/MinusP_3p125Sc_VBM_k8_b384_wfk.cube",
    "3.125Sc_WFK_Cube/MinusP_3p125Sc_CBM_k5_b385_wfk.cube",
    "3.125Sc_WFK_Cube/PlusP_3p125Sc_VBM_k1_b384_wfk.cube",
    "3.125Sc_WFK_Cube/PlusP_3p125Sc_CBM_k3_b385_wfk.cube",

    "6.25Sc_WFK_Cube/MinusP_6p25Sc_VBM_k7_b383_wfk.cube",
    "6.25Sc_WFK_Cube/MinusP_6p25Sc_CBM_k5_b384_wfk.cube",
    "6.25Sc_WFK_Cube/PlusP_6p25Sc_VBM_k1_b383_wfk.cube",
    "6.25Sc_WFK_Cube/PlusP_6p25Sc_CBM_k3_b384_wfk.cube",

    "9.375Sc_WFK_Cube/MinusP_9p375Sc_VBM_k4_b383_wfk.cube",
    "9.375Sc_WFK_Cube/MinusP_9p375Sc_CBM_k5_b384_wfk.cube",
    "9.375Sc_WFK_Cube/PlusP_9p375Sc_VBM_k1_b383_wfk.cube",
    "9.375Sc_WFK_Cube/PlusP_9p375Sc_CBM_k6_b384_wfk.cube",

    # Zr
    "3.125Zr_WFK_Cube/MinusP_3p125Zr_VBM_k7_b384_wfk.cube",
    "3.125Zr_WFK_Cube/MinusP_3p125Zr_CBM_k6_b385_wfk.cube",
    "3.125Zr_WFK_Cube/PlusP_3p125Zr_VBM_k1_b384_wfk.cube",
    "3.125Zr_WFK_Cube/PlusP_3p125Zr_CBM_k3_b385_wfk.cube",

    "6.25Zr_WFK_Cube/MinusP_6p25Zr_VBM_k7_b384_wfk.cube",
    "6.25Zr_WFK_Cube/MinusP_6p25Zr_CBM_k6_b385_wfk.cube",
    "6.25Zr_WFK_Cube/PlusP_6p25Zr_VBM_k1_b384_wfk.cube",
    "6.25Zr_WFK_Cube/PlusP_6p25Zr_CBM_k3_b385_wfk.cube",

    "9.375Zr_WFK_Cube/MinusP_9p375Zr_VBM_k7_b384_wfk.cube",
    "9.375Zr_WFK_Cube/MinusP_9p375Zr_CBM_k6_b385_wfk.cube",
    "9.375Zr_WFK_Cube/PlusP_9p375Zr_VBM_k1_b384_wfk.cube",
    "9.375Zr_WFK_Cube/PlusP_9p375Zr_CBM_k6_b385_wfk.cube",
]


def get_dopant(name):
    if "Sc" in name:
        return "Sc"
    if "Zr" in name:
        return "Zr"
    if "Y" in name:
        return "Y"
    raise ValueError(f"Cannot determine dopant from {name}")


def get_concentration(name):
    if "3p125" in name:
        return "3.125"
    if "6p25" in name:
        return "6.25"
    if "9p375" in name:
        return "9.375"
    raise ValueError(f"Cannot determine concentration from {name}")


def read_cube(path):
    with open(path, "r") as f:
        lines = f.readlines()

    c1, c2 = lines[0], lines[1]

    p = lines[2].split()
    natoms = int(float(p[0]))
    origin = np.array(list(map(float, p[1:4])))

    grid = []
    for i in range(3):
        q = lines[3 + i].split()
        grid.append((int(float(q[0])), np.array(list(map(float, q[1:4])))))

    atom_lines = lines[6:6 + abs(natoms)]
    data_lines = lines[6 + abs(natoms):]

    atoms = []
    for idx, line in enumerate(atom_lines):
        q = line.split()
        Z = int(float(q[0]))
        atoms.append({
            "index": idx + 1,
            "Z": Z,
            "elem": Z_TO_ELEM.get(Z, f"Z{Z}"),
            "charge": float(q[1]),
            "xyz": np.array(list(map(float, q[2:5]))),
        })

    values = np.array([float(x) for line in data_lines for x in line.split()])
    return c1, c2, origin, grid, atoms, values


def distance_ang(a, b):
    return np.linalg.norm(a - b) * BOHR_TO_ANG


def select_local_atoms(atoms, dopants, dopant):
    keep = {d["index"]: d for d in dopants}

    for atom in atoms:
        if atom["index"] in keep:
            continue

        dmin = min(distance_ang(atom["xyz"], d["xyz"]) for d in dopants)

        if atom["elem"] == "O" and dmin <= O_CUTOFF[dopant]:
            keep[atom["index"]] = atom

        elif atom["elem"] == "Hf" and dmin <= HF_CUTOFF[dopant]:
            keep[atom["index"]] = atom

    return sorted(keep.values(), key=lambda a: a["index"])


def crop_values(values, origin, grid, dopants, dopant):
    nx, vx = grid[0]
    ny, vy = grid[1]
    nz, vz = grid[2]

    radius = CROP_RADIUS[dopant]
    dopant_xyz = [d["xyz"] for d in dopants]

    cropped = np.zeros_like(values)
    kept = 0

    for i in range(nx):
        ri = origin + i * vx
        for j in range(ny):
            rij = ri + j * vy
            for k in range(nz):
                r = rij + k * vz
                dmin = min(distance_ang(r, dxyz) for dxyz in dopant_xyz)

                if dmin <= radius:
                    cropped[i, j, k] = values[i, j, k]
                    kept += 1

    return cropped, kept


def write_cube(path, c1, c2, origin, grid, atoms, values):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(c1)
        f.write(c2)

        f.write(
            f"{len(atoms):5d} "
            f"{origin[0]:12.6f} "
            f"{origin[1]:12.6f} "
            f"{origin[2]:12.6f}\n"
        )

        for n, vec in grid:
            f.write(
                f"{n:5d} "
                f"{vec[0]:12.6f} "
                f"{vec[1]:12.6f} "
                f"{vec[2]:12.6f}\n"
            )

        for atom in atoms:
            x, y, z = atom["xyz"]
            f.write(
                f"{atom['Z']:5d} {atom['charge']:12.6f} "
                f"{x:12.6f} {y:12.6f} {z:12.6f}\n"
            )

        flat = values.ravel()
        for i in range(0, len(flat), 6):
            f.write(" ".join(f"{v:13.5e}" for v in flat[i:i + 6]) + "\n")


def process_one(rel_file):
    src = INPUT_ROOT / rel_file

    if not src.exists():
        print(f"MISSING: {src}")
        return False

    fname = Path(rel_file).name
    dopant = get_dopant(fname)
    conc = get_concentration(fname)

    out_dir = OUTPUT_ROOT / f"{conc}{dopant}_WFK_Best"
    out_file = out_dir / f"Best_{fname}"

    c1, c2, origin, grid, atoms, values = read_cube(src)

    nx, _ = grid[0]
    ny, _ = grid[1]
    nz, _ = grid[2]
    expected = nx * ny * nz

    if values.size != expected:
        raise ValueError(f"{fname}: expected {expected}, got {values.size}")

    values = values.reshape((nx, ny, nz))

    dopants = [a for a in atoms if a["Z"] == DOPANT_Z[dopant]]
    if not dopants:
        raise ValueError(f"No {dopant} dopant found in {fname}")

    keep_atoms = select_local_atoms(atoms, dopants, dopant)
    cropped, kept_grid = crop_values(values, origin, grid, dopants, dopant)

    write_cube(out_file, c1, c2, origin, grid, keep_atoms, cropped)

    nonzero = cropped[np.abs(cropped) > 1e-12]

    print("=" * 90)
    print(f"Input : {src}")
    print(f"Output: {out_file}")
    print(f"System: {conc}% {dopant}")
    print(f"Dopant atoms found: {len(dopants)}")
    print(f"Atoms kept: {len(keep_atoms)}")
    print(f"Crop radius: {CROP_RADIUS[dopant]:.2f} Å")
    print(f"Grid kept: {kept_grid} / {expected}")

    if nonzero.size > 0:
        print(f"Data min: {nonzero.min():.6e}")
        print(f"Data max: {nonzero.max():.6e}")
    else:
        print("WARNING: cropped data is all zero.")

    return True


def main():
    print("=" * 90)
    print("Generating Best WFK cube files")
    print(f"Input root : {INPUT_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    print("=" * 90)

    if not INPUT_ROOT.exists():
        print(f"ERROR: input folder does not exist: {INPUT_ROOT}")
        return

    created = 0
    missing = 0
    failed = 0
    log_lines = []

    for rel_file in TARGET_FILES:
        try:
            ok = process_one(rel_file)

            if ok:
                created += 1
                log_lines.append(f"CREATED: {rel_file}")
            else:
                missing += 1
                log_lines.append(f"MISSING: {rel_file}")

        except Exception as e:
            failed += 1
            print(f"ERROR: {rel_file}: {e}")
            log_lines.append(f"ERROR: {rel_file} | {e}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_file = OUTPUT_ROOT / "9Cube_WFK_All_Best_log.txt"

    with open(log_file, "w") as f:
        f.write("9Cube_WFK_All_Best generation log\n")
        f.write("=" * 80 + "\n")
        f.write(f"Input root : {INPUT_ROOT}\n")
        f.write(f"Output root: {OUTPUT_ROOT}\n")
        f.write(f"Created    : {created}\n")
        f.write(f"Missing    : {missing}\n")
        f.write(f"Failed     : {failed}\n")
        f.write("=" * 80 + "\n\n")
        for line in log_lines:
            f.write(line + "\n")

    print("=" * 90)
    print("DONE")
    print(f"Created: {created}")
    print(f"Missing: {missing}")
    print(f"Failed : {failed}")
    print(f"Log    : {log_file}")
    print("=" * 90)


if __name__ == "__main__":
    main()
