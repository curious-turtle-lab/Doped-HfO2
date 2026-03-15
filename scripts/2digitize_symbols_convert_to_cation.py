#!/usr/bin/env python3

import re
import csv
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# 1) EXP data you provided (all-atom%)
# ----------------------------
SC_EXP_ALLATOM = np.array([
    [0.16, 16.25],
    [0.739, 21.30],
    [1.48, 11.85],
    [0.75, 21.20],
    [1.50, 11.00],
    [2.35, 6.00],
    [3.875, 3.00],
], dtype=float)

Y_EXP_ALLATOM = np.array([
    [0.74, 22.45],
    [1.47, 12.75],
    [0.80, 21.50],
    [1.60, 12.50],
    [2.86, 6.00],
], dtype=float)

# Zr EXP points (all-atom%, Psw)
Zr_EXP_ALLATOM = np.array([
    [0.625, 12.60],
    [6.49,  22.50],
], dtype=float)


def exp_allatom_to_cation(exp_allatom: np.ndarray) -> np.ndarray:
    """Return columns: [x_allatom, x_cation, Psw]."""
    x_all = exp_allatom[:, 0]
    psw = exp_allatom[:, 1]
    x_cat = 3.0 * x_all
    return np.column_stack([x_all, x_cat, psw])


# ----------------------------
# 2) Force branch overrides (edit here)
# ----------------------------
FORCE_BRANCH = {
    "Sc": {9.375: 0},
    "Y":  {9.375: 0},
    "Zr": {3.125: 1, 6.25: 1, 9.375: 1},
}


# ----------------------------
# 3) Parse Data-*.txt blocks
# ----------------------------
def parse_calc_blocks(path: Path):
    txt = path.read_text(errors="ignore")

    # New format:
    #   Sc doping (%)           : 3.125
    #   ...
    #   ΔP0     [µC/cm^2]       : -24.38
    #   ...
    #   Pq      [µC/cm^2]       : 15.90
    #
    # Also accepts Y/Zr in place of Sc.

    starts = [
        m.start()
        for m in re.finditer(
            r"(?:Sc|Y|Zr)\s+doping\s+\(%\)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
            txt
        )
    ]

    if not starts:
        raise ValueError(
            f"Could not parse any calc blocks from: {path}\n"
            f"Expected blocks like 'Sc doping (%) : 3.125' or 'Y doping (%) : 3.125'."
        )

    starts.append(len(txt))
    out = []

    for i in range(len(starts) - 1):
        chunk = txt[starts[i]:starts[i + 1]]

        m_dop = re.search(
            r"(?:Sc|Y|Zr)\s+doping\s+\(%\)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
            chunk
        )
        if not m_dop:
            continue
        doping_cation = float(m_dop.group(1))

        m_dp0 = re.search(
            r"ΔP0\s+\[µC/cm\^2\]\s*:\s*([+\-]?[0-9]+(?:\.[0-9]+)?)",
            chunk
        )
        if not m_dp0:
            raise ValueError(f"Missing ΔP0 [µC/cm^2] for {doping_cation}% in {path}")
        dp0_uc = float(m_dp0.group(1))

        m_pq = re.search(
            r"Pq\s+\[µC/cm\^2\]\s*:\s*([+\-]?[0-9]+(?:\.[0-9]+)?)",
            chunk
        )
        if not m_pq:
            raise ValueError(f"Missing Pq [µC/cm^2] for {doping_cation}% in {path}")
        pq_uc = float(m_pq.group(1))

        out.append({
            "doping_cation": doping_cation,
            "dp0_uc": dp0_uc,
            "pq_uc": pq_uc,
        })

    out.sort(key=lambda d: d["doping_cation"])

    if not out:
        raise ValueError(f"Parsed zero blocks from: {path}")

    return out


# ----------------------------
# 4) Branch selection
# ----------------------------
def choose_branch_best_match(system_name: str,
                             xcat: float,
                             dp0_uc: float,
                             pq_uc: float,
                             exp_xcat: np.ndarray,
                             exp_psw: np.ndarray,
                             nmin: int = -12,
                             nmax: int = 12):
    xkey = round(float(xcat), 3)
    forced_map = FORCE_BRANCH.get(system_name, {})
    if xkey in forced_map:
        n = int(forced_map[xkey])
        return n, float(abs(dp0_uc + n * pq_uc)), "forced"

    j = int(np.argmin(np.abs(exp_xcat - xcat)))
    target = float(exp_psw[j])

    ns = np.arange(nmin, nmax + 1, dtype=int)
    psw_all = np.abs(dp0_uc + ns * pq_uc)
    k = int(np.argmin(np.abs(psw_all - target)))
    n = int(ns[k])
    return n, float(psw_all[k]), "best-match"


# ----------------------------
# 5) Table helpers + CSV export
# ----------------------------
def build_table(exp_sc, exp_y, exp_zr, sc_rows, y_rows, zr_rows):
    headers = ["System", "Type", "all-atom%", "cation%", "Psw (µC/cm²)", "n"]
    rows = []

    for xall, xcat, psw in exp_sc:
        rows.append(["Sc", "EXP", f"{xall:.3f}", f"{xcat:.3f}", f"{psw:.2f}", "—"])
    for xall, xcat, psw in exp_y:
        rows.append(["Y", "EXP", f"{xall:.3f}", f"{xcat:.3f}", f"{psw:.2f}", "—"])
    for xall, xcat, psw in exp_zr:
        rows.append(["Zr", "EXP", f"{xall:.3f}", f"{xcat:.3f}", f"{psw:.2f}", "—"])

    for r in sc_rows:
        rows.append(["Sc", "CALC", f"{r['xall']:.3f}", f"{r['xcat']:.3f}", f"{r['psw']:.2f}", f"{r['n']:+d}"])
    for r in y_rows:
        rows.append(["Y", "CALC", f"{r['xall']:.3f}", f"{r['xcat']:.3f}", f"{r['psw']:.2f}", f"{r['n']:+d}"])
    for r in zr_rows:
        rows.append(["Zr", "CALC", f"{r['xall']:.3f}", f"{r['xcat']:.3f}", f"{r['psw']:.2f}", f"{r['n']:+d}"])

    def keyfun(rr):
        sys_order = {"Sc": 0, "Y": 1, "Zr": 2}
        typ = 0 if rr[1] == "EXP" else 1
        xcat = float(rr[3])
        return (sys_order.get(rr[0], 99), typ, xcat)

    rows.sort(key=keyfun)
    return headers, rows


def export_csv(out_csv: Path, headers, rows):
    """
    Export a paper-ready CSV (Excel/Word friendly).
    """
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def add_table(ax, headers, rows, fontsize=10, yscale=1.45):
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=headers, loc="center",
                   cellLoc="center", colLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1.0, yscale)
    return tbl


# ----------------------------
# 6) Plot styling + legend
# ----------------------------
S_EXP = 220
S_CALC = 260

X_JITTER = {
    "Sc EXP":  -0.06,
    "Y EXP":   +0.06,
    "Zr EXP":  +0.00,
    "Sc CALC": -0.05,
    "Y CALC":  +0.05,
    "Zr CALC": +0.00,
}


def compute_xlim_with_padding(all_x: np.ndarray):
    xmin = float(np.min(all_x))
    xmax = float(np.max(all_x))
    return xmin - 0.55, xmax + 0.85


def annotate_n(ax, x, y, n, xlim):
    xmin, xmax = xlim
    near_right = x > (xmax - 0.85)
    if near_right:
        ax.annotate(f"n={n:+d}", (x, y), xytext=(-10, 6), textcoords="offset points",
                    ha="right", va="bottom", fontsize=12, clip_on=False)
    else:
        ax.annotate(f"n={n:+d}", (x, y), xytext=(10, 6), textcoords="offset points",
                    ha="left", va="bottom", fontsize=12, clip_on=False)


def add_clean_legend(ax):
    leg = ax.legend(
        loc="lower right",          # <-- bottom-right
        bbox_to_anchor=(0.98, 0.02),# <-- small inset from the corner
        framealpha=0.95,
        fontsize=14,
        markerscale=0.85,
        handlelength=2.4,
        handletextpad=0.9,
        labelspacing=0.8,
        borderpad=0.7,
        borderaxespad=0.0,
    )
    leg.get_frame().set_linewidth(1.2)
    return leg


def save_fig(fig, outpath: Path):
    fig.savefig(outpath, dpi=260, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def make_plot_only(exp_sc, exp_y, exp_zr, sc_rows, y_rows, zr_rows, outpath: Path):
    fig, ax = plt.subplots(figsize=(14.5, 8.2))

    ax.scatter(exp_sc[:, 1] + X_JITTER["Sc EXP"], exp_sc[:, 2],
               marker="o", s=S_EXP, label="Sc EXP (cation%)")
    ax.scatter(exp_y[:, 1] + X_JITTER["Y EXP"], exp_y[:, 2],
               marker="^", s=S_EXP, label="Y EXP (cation%)")
    ax.scatter(exp_zr[:, 1] + X_JITTER["Zr EXP"], exp_zr[:, 2],
               marker="X", s=S_EXP, label="Zr EXP (cation%)")

    sc_x = np.array([r["xcat"] for r in sc_rows]) + X_JITTER["Sc CALC"]
    sc_y = np.array([r["psw"] for r in sc_rows])
    sc_n = [r["n"] for r in sc_rows]
    ax.scatter(sc_x, sc_y, marker="s", s=S_CALC, label="Sc CALC (branch)")

    y_x = np.array([r["xcat"] for r in y_rows]) + X_JITTER["Y CALC"]
    y_y = np.array([r["psw"] for r in y_rows])
    y_n = [r["n"] for r in y_rows]
    ax.scatter(y_x, y_y, marker="D", s=S_CALC, label="Y CALC (branch)")

    zr_x = np.array([r["xcat"] for r in zr_rows]) + X_JITTER["Zr CALC"]
    zr_y = np.array([r["psw"] for r in zr_rows])
    zr_n = [r["n"] for r in zr_rows]
    ax.scatter(zr_x, zr_y, marker="P", s=S_CALC, label="Zr CALC (branch)")

    all_x = np.concatenate([
        exp_sc[:, 1], exp_y[:, 1], exp_zr[:, 1],
        np.array([r["xcat"] for r in sc_rows]),
        np.array([r["xcat"] for r in y_rows]),
        np.array([r["xcat"] for r in zr_rows]),
    ])
    xlim = compute_xlim_with_padding(all_x)
    ax.set_xlim(*xlim)

    for x, yv, n in zip(sc_x, sc_y, sc_n):
        annotate_n(ax, float(x), float(yv), int(n), xlim)
    for x, yv, n in zip(y_x, y_y, y_n):
        annotate_n(ax, float(x), float(yv), int(n), xlim)
    for x, yv, n in zip(zr_x, zr_y, zr_n):
        annotate_n(ax, float(x), float(yv), int(n), xlim)

    ax.set_title("Y-, Sc-, and Zr-doped HfO₂: EXP vs CALC (Berry branch)")
    ax.set_xlabel("Cation doping (%)   (x_cation = 3 × x_all-atom)")
    ax.set_ylabel(r"$P_{sw}$  ($\mu$C/cm$^2$)")
    ax.grid(True, alpha=0.35)

    add_clean_legend(ax)
    save_fig(fig, outpath)


def make_plot_plus_table(exp_sc, exp_y, exp_zr, sc_rows, y_rows, zr_rows, headers, rows, outpath: Path):
    fig = plt.figure(figsize=(14.5, 9.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.6], hspace=0.18)

    ax = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[1, 0])

    ax.scatter(exp_sc[:, 1] + X_JITTER["Sc EXP"], exp_sc[:, 2],
               marker="o", s=S_EXP, label="Sc EXP (cation%)")
    ax.scatter(exp_y[:, 1] + X_JITTER["Y EXP"], exp_y[:, 2],
               marker="^", s=S_EXP, label="Y EXP (cation%)")
    ax.scatter(exp_zr[:, 1] + X_JITTER["Zr EXP"], exp_zr[:, 2],
               marker="X", s=S_EXP, label="Zr EXP (cation%)")

    sc_x = np.array([r["xcat"] for r in sc_rows]) + X_JITTER["Sc CALC"]
    sc_y = np.array([r["psw"] for r in sc_rows])
    sc_n = [r["n"] for r in sc_rows]
    ax.scatter(sc_x, sc_y, marker="s", s=S_CALC, label="Sc CALC (branch)")

    y_x = np.array([r["xcat"] for r in y_rows]) + X_JITTER["Y CALC"]
    y_y = np.array([r["psw"] for r in y_rows])
    y_n = [r["n"] for r in y_rows]
    ax.scatter(y_x, y_y, marker="D", s=S_CALC, label="Y CALC (branch)")

    zr_x = np.array([r["xcat"] for r in zr_rows]) + X_JITTER["Zr CALC"]
    zr_y = np.array([r["psw"] for r in zr_rows])
    zr_n = [r["n"] for r in zr_rows]
    ax.scatter(zr_x, zr_y, marker="P", s=S_CALC, label="Zr CALC (branch)")

    all_x = np.concatenate([
        exp_sc[:, 1], exp_y[:, 1], exp_zr[:, 1],
        np.array([r["xcat"] for r in sc_rows]),
        np.array([r["xcat"] for r in y_rows]),
        np.array([r["xcat"] for r in zr_rows]),
    ])
    xlim = compute_xlim_with_padding(all_x)
    ax.set_xlim(*xlim)

    for x, yv, n in zip(sc_x, sc_y, sc_n):
        annotate_n(ax, float(x), float(yv), int(n), xlim)
    for x, yv, n in zip(y_x, y_y, y_n):
        annotate_n(ax, float(x), float(yv), int(n), xlim)
    for x, yv, n in zip(zr_x, zr_y, zr_n):
        annotate_n(ax, float(x), float(yv), int(n), xlim)

    ax.set_title("Y-, Sc-, and Zr-doped HfO₂: EXP vs CALC (Berry branch)")
    ax.set_xlabel("Cation doping (%)   (x_cation = 3 × x_all-atom)")
    ax.set_ylabel(r"$P_{sw}$  ($\mu$C/cm$^2$)")
    ax.grid(True, alpha=0.35)

    add_clean_legend(ax)
    add_table(ax_tbl, headers, rows, fontsize=10, yscale=1.45)

    save_fig(fig, outpath)


def make_table_only(headers, rows, outpath: Path):
    fig, ax = plt.subplots(figsize=(14.5, 5.6))
    add_table(ax, headers, rows, fontsize=10, yscale=1.45)
    fig.savefig(outpath, dpi=260, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# ----------------------------
# 7) Main
# ----------------------------
def main():
    work = Path.cwd()

    sc_path = work / "Data-Sc.txt"
    y_path = work / "Data-Y.txt"
    zr_path = work / "Data-Zr.txt"

    for p in (sc_path, y_path, zr_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    exp_sc = exp_allatom_to_cation(SC_EXP_ALLATOM)
    exp_y = exp_allatom_to_cation(Y_EXP_ALLATOM)
    exp_zr = exp_allatom_to_cation(Zr_EXP_ALLATOM)

    sc_blocks = parse_calc_blocks(sc_path)
    y_blocks = parse_calc_blocks(y_path)
    zr_blocks = parse_calc_blocks(zr_path)

    sc_rows = []
    for b in sc_blocks:
        xcat = float(b["doping_cation"])
        xall = xcat / 3.0
        n, psw, _ = choose_branch_best_match("Sc", xcat, b["dp0_uc"], b["pq_uc"], exp_sc[:, 1], exp_sc[:, 2])
        sc_rows.append({"xcat": xcat, "xall": xall, "n": n, "psw": psw})
    sc_rows.sort(key=lambda r: r["xcat"])

    y_rows = []
    for b in y_blocks:
        xcat = float(b["doping_cation"])
        xall = xcat / 3.0
        n, psw, _ = choose_branch_best_match("Y", xcat, b["dp0_uc"], b["pq_uc"], exp_y[:, 1], exp_y[:, 2])
        y_rows.append({"xcat": xcat, "xall": xall, "n": n, "psw": psw})
    y_rows.sort(key=lambda r: r["xcat"])

    zr_rows = []
    for b in zr_blocks:
        xcat = float(b["doping_cation"])
        xall = xcat / 3.0
        n, psw, _ = choose_branch_best_match("Zr", xcat, b["dp0_uc"], b["pq_uc"], exp_zr[:, 1], exp_zr[:, 2])
        zr_rows.append({"xcat": xcat, "xall": xall, "n": n, "psw": psw})
    zr_rows.sort(key=lambda r: r["xcat"])

    headers, table_rows = build_table(exp_sc, exp_y, exp_zr, sc_rows, y_rows, zr_rows)

    # ---- EXPORT CSV (paper-ready)
    out_csv = work / "EXP_vs_CALC_Sc_Y_Zr_table.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(table_rows)

    # ---- figures
    make_plot_only(exp_sc, exp_y, exp_zr, sc_rows, y_rows, zr_rows,
                   work / "EXP_vs_CALC_Sc_Y_Zr_plot_only.png")
    """               
    make_plot_plus_table(exp_sc, exp_y, exp_zr, sc_rows, y_rows, zr_rows,
                         headers, table_rows,
                         work / "EXP_vs_CALC_Sc_Y_Zr_plot_plus_table.png")
    make_table_only(headers, table_rows,
                    work / "EXP_vs_CALC_Sc_Y_Zr_table_only.png")
    """
    print("[OK] Wrote:")
#    print(" ", out_csv.resolve())
    print(" ", (work / "EXP_vs_CALC_Sc_Y_Zr_plot_only.png").resolve())
#    print(" ", (work / "EXP_vs_CALC_Sc_Y_Zr_plot_plus_table.png").resolve())
#    print(" ", (work / "EXP_vs_CALC_Sc_Y_Zr_table_only.png").resolve())


if __name__ == "__main__":
    main()

