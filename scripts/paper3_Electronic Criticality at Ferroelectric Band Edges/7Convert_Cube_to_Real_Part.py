#!/usr/bin/env python3
from pathlib import Path
import re
import subprocess

CUBE_DIR = Path("./cubes")
OUT_ROOT = Path("./CubeReal")
CUT3D = "cut3d"

SYSTEMS = {
    "3p125Sc": {
        "folder": "3.125Sc_Cube_Real",
        "MinusP": "/home/yanan/Material/Y_dope_project/3.125Per-Sc-HfO2-2x2x2/berry-scf-Results/berry_MinusP_3.125Sc_HfO2_12144332/berry_MinusP_3o_WFK.nc",
        "PlusP":  "/home/yanan/Material/Y_dope_project/3.125Per-Sc-HfO2-2x2x2/berry-scf-Results/berry_PlusP_3.125Sc_HfO2_12144296/berry_PlusP_3o_WFK.nc",
    },
    "6p25Sc": {
        "folder": "6.25Sc_Cube_Real",
        "MinusP": "/home/yanan/Material/Y_dope_project/6.25Per-Sc-HfO2-2x2x2/berry-scf-results/berry_MinusP_6.25Sc_HfO2_12144457/berry_MinusP_6o_WFK.nc",
        "PlusP":  "/home/yanan/Material/Y_dope_project/6.25Per-Sc-HfO2-2x2x2/berry-scf-results/berry_PlusP_6.25Sc_HfO2_12144349/berry_PlusP_6o_WFK.nc",
    },
    "9p375Sc": {
        "folder": "9.375Sc_Cube_Real",
        "MinusP": "/home/yanan/Material/Y_dope_project/9.375Sc-HfO2-2x2x2/berry-scf-Results/berry_MinusP_9.375Sc_HfO2_12144529/berry_MinusP_9o_WFK.nc",
        "PlusP":  "/home/yanan/Material/Y_dope_project/9.375Sc-HfO2-2x2x2/berry-scf-Results/berry_PlusP_9.375Sc_HfO2_12144507/berry_PlusP_9o_WFK.nc",
    },

    "3p125Y": {
        "folder": "3.125Y_Cube_Real",
        "MinusP": "/home/yanan/Material/Y_dope_project/3.125Per-Y-HfO2-2x2x2/berry-scf-results/berry_MinusP_3.125Per_Y_HfO2_12118864/berry_MinusP_3o_WFK.nc",
        "PlusP":  "/home/yanan/Material/Y_dope_project/3.125Per-Y-HfO2-2x2x2/berry-scf-results/berry_PlusP_3.125Per_Y_HfO2_12118862/berry_PlusP_3o_WFK.nc",
    },
    "6p25Y": {
        "folder": "6.25Y_Cube_Real",
        "MinusP": "/home/yanan/Material/Y_dope_project/6.25Per-Y-HfO2-2x2x2/berry-scf-results/berry_MinusP_6.25Y_HfO2_12397618/berry_MinusP_6o_WFK.nc",
        "PlusP":  "/home/yanan/Material/Y_dope_project/6.25Per-Y-HfO2-2x2x2/berry-scf-results/berry_PlusP_6.25Y_HfO2_12651723/berry_PlusP_6o_WFK.nc",
    },
    "9p375Y": {
        "folder": "9.375Y_Cube_Real",
        "MinusP": "/home/yanan/Material/Y_dope_project/9.375Y-HfO2-2x2x2/9.375Y-scf-berry-2x2x2-Results/berry_MinusP_9.375Y_HfO2_12121996/berry_MinusP_9o_WFK.nc",
        "PlusP":  "/home/yanan/Material/Y_dope_project/9.375Y-HfO2-2x2x2/9.375Y-scf-berry-2x2x2-Results/berry_PlusP_9.375Y_HfO2_12122002/berry_PlusP_9o_WFK.nc",
    },

    "3p125Zr": {
        "folder": "3.125Zr_Cube_Real",
        "MinusP": "/home/yanan/Material/Y_dope_project/3.125Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_MinusP_3.125Zr_HfO2_12247642/berry_MinusP_3o_WFK.nc",
        "PlusP":  "/home/yanan/Material/Y_dope_project/3.125Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_PlusP_3.125Zr_HfO2_12247641/berry_PlusP_3o_WFK.nc",
    },
    "6p25Zr": {
        "folder": "6.25Zr_Cube_Real",
        "MinusP": "/home/yanan/Material/Y_dope_project/6.25Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_MinusP_6.25Zr_HfO2_12248602/berry_MinusP_6o_WFK.nc",
        "PlusP":  "/home/yanan/Material/Y_dope_project/6.25Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_PlusP_6.25Zr_HfO2_12248413/berry_PlusP_6o_WFK.nc",
    },
    "9p375Zr": {
        "folder": "9.375Zr_Cube_Real",
        "MinusP": "/home/yanan/Material/Y_dope_project/9.375Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_MinusP_9.375Zr_HfO2_12247637/berry_MinusP_9o_WFK.nc",
        "PlusP":  "/home/yanan/Material/Y_dope_project/9.375Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_PlusP_9.375Zr_HfO2_12247629/berry_PlusP_9o_WFK.nc",
    },
}


def parse_cube_name(name: str):
    m = re.match(r"(MinusP|PlusP)_(\d+p\d+(?:Sc|Y|Zr))_(VBM|CBM)_k(\d+)_b(\d+)\.cube$", name)
    if not m:
        return None
    state, tag, edge, k, band = m.groups()
    return state, tag, edge, int(k), int(band)


def cut3d_real_cube(wfk: Path, k: int, band: int, out_cube: Path, log_file: Path):
    out_cube.parent.mkdir(parents=True, exist_ok=True)

    # No leading blank line. This is the main fix.
    cut3d_input = "\n".join([
        str(wfk),
        str(k),
        str(band),
        "0",          # no GW wavefunction
        "0",          # no atomic analysis
        "2",          # 3D formatted real data
        str(out_cube)
    ]) + "\n"

    result = subprocess.run(
        [CUT3D],
        input=cut3d_input,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    log_file.write_text(result.stdout)

    if result.returncode != 0:
        print(f"  FAILED: cut3d returned {result.returncode}")
        print(f"  Log: {log_file}")
        return False

    if not out_cube.exists():
        print(f"  WARNING: cut3d finished but output not found: {out_cube}")
        print(f"  Log: {log_file}")
        return False

    print(f"  OK: {out_cube}")
    return True


def main():
    if not CUBE_DIR.exists():
        raise SystemExit(f"Cannot find cube folder: {CUBE_DIR}")

    OUT_ROOT.mkdir(exist_ok=True)

    cubes = sorted(CUBE_DIR.glob("*.cube"))
    if not cubes:
        raise SystemExit(f"No .cube files found in {CUBE_DIR}")

    ok = 0
    failed = 0
    skipped = 0

    for cube in cubes:
        parsed = parse_cube_name(cube.name)

        if parsed is None:
            skipped += 1
            continue

        state, tag, edge, k, band = parsed

        if tag not in SYSTEMS:
            print(f"SKIP unknown tag: {cube.name}")
            skipped += 1
            continue

        wfk = Path(SYSTEMS[tag][state])
        out_dir = OUT_ROOT / SYSTEMS[tag]["folder"]
        out_cube = out_dir / cube.name.replace(".cube", "_Realdata.cube")
        log_file = out_dir / cube.name.replace(".cube", "_cut3d.log")

        print(f"\nProcessing: {cube.name}")
        print(f"  WFK : {wfk}")
        print(f"  k/b : k={k}, band={band}")
        print(f"  OUT : {out_cube}")

        if not wfk.exists():
            print(f"  FAILED: WFK file does not exist: {wfk}")
            failed += 1
            continue

        success = cut3d_real_cube(wfk, k, band, out_cube, log_file)
        if success:
            ok += 1
        else:
            failed += 1

    summary = OUT_ROOT / "convert_summary.txt"
    summary.write_text(
        f"Completed real cube conversion\n"
        f"OK      : {ok}\n"
        f"FAILED  : {failed}\n"
        f"SKIPPED : {skipped}\n"
    )

    print("\n========================================")
    print(f"OK      : {ok}")
    print(f"FAILED  : {failed}")
    print(f"SKIPPED : {skipped}")
    print(f"Summary : {summary}")
    print("========================================")


if __name__ == "__main__":
    main()
