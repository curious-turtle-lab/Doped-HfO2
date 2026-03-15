#!/usr/bin/env python3
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# ============================================================
# Zr-doped HfO2 Berry-phase output files
# ============================================================
CASES = [
    {
        "doping_percent": 3.125,
        "plus_out": Path("/home/yanan/Material/Y_dope_project/3.125Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_PlusP_3.125Zr_HfO2_12247641/berry_PlusP_3.125Per_Zr_HfO2_2x2x2.out"),
        "minus_out": Path("/home/yanan/Material/Y_dope_project/3.125Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_MinusP_3.125Zr_HfO2_12247642/berry_MinusP_3.125Per_Zr_HfO2_2x2x2.out"),
    },
    {
        "doping_percent": 6.25,
        "plus_out": Path("/home/yanan/Material/Y_dope_project/6.25Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_PlusP_6.25Zr_HfO2_12248413/berry_PlusP_6.25Per_Zr_HfO2_2x2x2.out"),
        "minus_out": Path("/home/yanan/Material/Y_dope_project/6.25Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_MinusP_6.25Zr_HfO2_12248602/berry_MinusP_6.25Per_Zr_HfO2_2x2x2.out"),
    },
    {
        "doping_percent": 9.375,
        "plus_out": Path("/home/yanan/Material/Y_dope_project/9.375Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_PlusP_9.375Zr_HfO2_12247629/berry_PlusP_9.375Per_Zr_HfO2_2x2x2.out"),
        "minus_out": Path("/home/yanan/Material/Y_dope_project/9.375Per-Zr-HfO2-2x2x2/berry-scf-Results/berry_MinusP_9.375Zr_HfO2_12247637/berry_MinusP_9.375Per_Zr_HfO2_2x2x2.out"),
    },
]

OUTPUT_TXT = Path("Data-Zr.txt")

# =========================
# Constants
# =========================
E_CHARGE = 1.602176634e-19
BOHR_M = 0.529177210903e-10
ANG_M = 1e-10

FLOAT = r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[EeDd][-+]?\d+)?"
FLOAT_RE = re.compile(FLOAT)


def _to_float(tok: str) -> float:
    return float(tok.replace("D", "E").replace("d", "e"))


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    return path.read_text(errors="replace")


def parse_rfdir(text: str) -> Optional[Tuple[int, int, int]]:
    m = re.search(r"\brfdir\b\s+([01])\s+([01])\s+([01])", text)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def parse_last_polarization_scalar_Cm2(text: str) -> float:
    pat = re.compile(rf"\bPolarization\b\s+({FLOAT})\s+C/m\^2")
    ms = list(pat.finditer(text))
    if not ms:
        raise RuntimeError("No 'Polarization ... C/m^2' line found.")
    return _to_float(ms[-1].group(1))


def parse_rprimd_ang_from_out(text: str) -> Optional[List[List[float]]]:
    lines = text.splitlines()

    def last3(line: str) -> Optional[List[float]]:
        nums = [_to_float(x) for x in FLOAT_RE.findall(line)]
        if len(nums) < 3:
            return None
        return nums[-3:]

    for i, ln in enumerate(lines):
        if re.search(r"\brprimd\b", ln):
            vecs_bohr: List[List[float]] = []
            for j in range(1, 4):
                if i + j >= len(lines):
                    break
                v = last3(lines[i + j])
                if v is None:
                    break
                vecs_bohr.append(v)
            if len(vecs_bohr) == 3:
                return [[x * BOHR_M / ANG_M for x in row] for row in vecs_bohr]
    return None


def volume_from_rprimd_ang(rprimd_ang: List[List[float]]) -> float:
    ax, ay, az = [v * ANG_M for v in rprimd_ang[0]]
    bx, by, bz = [v * ANG_M for v in rprimd_ang[1]]
    cx, cy, cz = [v * ANG_M for v in rprimd_ang[2]]
    vol = (
        ax * (by * cz - bz * cy)
        - ay * (bx * cz - bz * cx)
        + az * (bx * cy - by * cx)
    )
    return abs(vol)


def vec_norm(v: Tuple[float, float, float]) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def polarization_quantum_Cm2(volume_m3: float, direction_vec_m: Tuple[float, float, float]) -> float:
    L = vec_norm(direction_vec_m)
    return E_CHARGE * L / volume_m3


def pick_axis_index(rfdir: Tuple[int, int, int]) -> int:
    if rfdir in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        return rfdir.index(1)
    return max(i for i, v in enumerate(rfdir) if v == 1)


def analyze_case(case: Dict) -> Dict:
    plus_text = read_text(case["plus_out"])
    minus_text = read_text(case["minus_out"])

    p_plus = parse_last_polarization_scalar_Cm2(plus_text)
    p_minus = parse_last_polarization_scalar_Cm2(minus_text)

    rfdir = parse_rfdir(plus_text)
    if rfdir is None:
        rfdir = parse_rfdir(minus_text)
    if rfdir is None:
        raise RuntimeError(f"Could not find rfdir for doping = {case['doping_percent']}%")

    idir = pick_axis_index(rfdir)
    axis_name = ["x/a", "y/b", "z/c"][idir]

    rprimd_ang = parse_rprimd_ang_from_out(plus_text)
    if rprimd_ang is None:
        raise RuntimeError(f"Could not parse rprimd from {case['plus_out']}")

    vol = volume_from_rprimd_ang(rprimd_ang)
    dir_vec_m = tuple(v * ANG_M for v in rprimd_ang[idir])
    p_quantum = polarization_quantum_Cm2(vol, dir_vec_m)

    delta_p0 = p_plus - p_minus
    ps0 = 0.5 * abs(delta_p0)

    return {
        "doping_percent": case["doping_percent"],
        "rfdir": rfdir,
        "axis_name": axis_name,
        "p_plus_cm2": p_plus,
        "p_minus_cm2": p_minus,
        "delta_p0_cm2": delta_p0,
        "ps0_cm2": ps0,
        "p_plus_ucm2": p_plus * 100.0,
        "p_minus_ucm2": p_minus * 100.0,
        "delta_p0_ucm2": delta_p0 * 100.0,
        "ps0_ucm2": ps0 * 100.0,
        "p_quantum_cm2": p_quantum,
        "p_quantum_ucm2": p_quantum * 100.0,
        "volume_m3": vol,
        "rprimd_ang": rprimd_ang,
        "plus_out": str(case["plus_out"]),
        "minus_out": str(case["minus_out"]),
    }


def write_output(results: List[Dict], outfile: Path) -> None:
    with outfile.open("w", encoding="utf-8") as f:
        f.write("Zr-doped HfO2 spontaneous polarization summary\n")
        f.write("=" * 90 + "\n\n")

        f.write("Compact table\n")
        f.write("-" * 90 + "\n")
        f.write(
            f"{'Zr(%)':>8}  {'rfdir':>10}  {'Axis':>6}  "
            f"{'P(+P)':>12}  {'P(-P)':>12}  {'ΔP0':>12}  "
            f"{'Ps(no-branch)':>16}  {'Pq':>10}\n"
        )
        f.write(
            f"{'':>8}  {'':>10}  {'':>6}  "
            f"{'(µC/cm²)':>12}  {'(µC/cm²)':>12}  {'(µC/cm²)':>12}  "
            f"{'(µC/cm²)':>16}  {'(µC/cm²)':>10}\n"
        )
        f.write("-" * 90 + "\n")

        for r in results:
            f.write(
                f"{r['doping_percent']:8.3f}  "
                f"{str(r['rfdir']):>10}  "
                f"{r['axis_name']:>6}  "
                f"{r['p_plus_ucm2']:12.2f}  "
                f"{r['p_minus_ucm2']:12.2f}  "
                f"{r['delta_p0_ucm2']:12.2f}  "
                f"{r['ps0_ucm2']:16.2f}  "
                f"{r['p_quantum_ucm2']:10.2f}\n"
            )

        f.write("\n")
        f.write("Detailed block\n")
        f.write("-" * 90 + "\n")

        for r in results:
            f.write(f"Zr doping (%)           : {r['doping_percent']:.3f}\n")
            f.write(f"Plus output             : {r['plus_out']}\n")
            f.write(f"Minus output            : {r['minus_out']}\n")
            f.write(f"rfdir                   : {r['rfdir']}\n")
            f.write(f"Polarization direction  : {r['axis_name']}\n")
            f.write(f"P(+P)   [C/m^2]         : {r['p_plus_cm2']:.12f}\n")
            f.write(f"P(-P)   [C/m^2]         : {r['p_minus_cm2']:.12f}\n")
            f.write(f"ΔP0     [C/m^2]         : {r['delta_p0_cm2']:.12f}\n")
            f.write(f"Ps0     [C/m^2]         : {r['ps0_cm2']:.12f}\n")
            f.write(f"P(+P)   [µC/cm^2]       : {r['p_plus_ucm2']:.2f}\n")
            f.write(f"P(-P)   [µC/cm^2]       : {r['p_minus_ucm2']:.2f}\n")
            f.write(f"ΔP0     [µC/cm^2]       : {r['delta_p0_ucm2']:.2f}\n")
            f.write(f"Ps0     [µC/cm^2]       : {r['ps0_ucm2']:.2f}\n")
            f.write(f"Pq      [C/m^2]         : {r['p_quantum_cm2']:.12f}\n")
            f.write(f"Pq      [µC/cm^2]       : {r['p_quantum_ucm2']:.2f}\n")
            f.write(f"Volume   [m^3]          : {r['volume_m3']:.6e}\n")
            f.write("rprimd (Angstrom)       :\n")
            for row in r["rprimd_ang"]:
                f.write("    " + "  ".join(f"{x:.6f}" for x in row) + "\n")
            f.write("-" * 90 + "\n")


def main() -> int:
    results = []
    for case in CASES:
        results.append(analyze_case(case))

    results.sort(key=lambda x: x["doping_percent"])
    write_output(results, OUTPUT_TXT)

    print(f"Wrote: {OUTPUT_TXT.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
