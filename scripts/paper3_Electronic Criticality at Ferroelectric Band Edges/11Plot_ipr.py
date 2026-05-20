# ============================================================
# PAPER-STYLE IPR FIGURE LIKE ATTACHED EXAMPLE
# ============================================================
"""
python3 11Plot_ipr.py

"""

def plot_ipr_paper_style(df):
    """
    Generate two-panel figure:
    (a) VBM IPR
    (b) CBM IPR

    Similar style to attached paper figure.
    X-axis = system index instead of time.
    """
    d = df.copy()

    # Convert IPR to 10^-3 scale like attached figure
    d["IPR_1e3"] = d["IPR"] * 1e3

    # Define clean case label
    d["case_label"] = (
        d["doping_percent"].astype(str) + "% "
        + d["dopant"].astype(str) + " "
        + d["polarization"].astype(str)
    )

    order_cols = ["dopant", "doping_percent", "polarization"]
    d = d.sort_values(order_cols)

    vbm = d[d["orbital"] == "VBM"].reset_index(drop=True)
    cbm = d[d["orbital"] == "CBM"].reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=False)

    # -------------------------
    # (a) VBM
    # -------------------------
    ax = axes[0]

    x = np.arange(len(vbm))
    y = vbm["IPR_1e3"].values

    ax.fill_between(x, y, alpha=0.45, label="VBM")
    ax.plot(x, y, linewidth=2)
    ax.bar(x, y, alpha=0.65)

    ax.text(0.03, 0.92, "(a) VBM", transform=ax.transAxes,
            fontsize=16, fontweight="bold")

    ax.set_ylabel(r"IPR $(10^{-3})$", fontsize=14)
    ax.set_xlabel("Doped HfO$_2$ cases", fontsize=14)
    ax.set_title("VBM localization", fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels(vbm["case_label"], rotation=90, fontsize=8)

    ax.legend(frameon=False, fontsize=11)

    # -------------------------
    # (b) CBM
    # -------------------------
    ax = axes[1]

    x = np.arange(len(cbm))
    y = cbm["IPR_1e3"].values

    ax.fill_between(x, y, alpha=0.45, label="CBM")
    ax.plot(x, y, linewidth=2)
    ax.bar(x, y, alpha=0.65)

    ax.text(0.03, 0.92, "(b) CBM", transform=ax.transAxes,
            fontsize=16, fontweight="bold")

    ax.set_ylabel(r"IPR $(10^{-3})$", fontsize=14)
    ax.set_xlabel("Doped HfO$_2$ cases", fontsize=14)
    ax.set_title("CBM localization", fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels(cbm["case_label"], rotation=90, fontsize=8)

    ax.legend(frameon=False, fontsize=11)

    plt.tight_layout()

    out_png = os.path.join(OUT_DIR, "Fig_IPR_paper_style_VBM_CBM.png")
    out_pdf = os.path.join(OUT_DIR, "Fig_IPR_paper_style_VBM_CBM.pdf")

    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print("Saved paper-style IPR figure:")
    print(out_png)
    print(out_pdf)
