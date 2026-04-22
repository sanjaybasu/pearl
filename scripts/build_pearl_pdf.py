#!/usr/bin/env python3
"""
Build a combined PDF of PEARL manuscript + supplement.
Steps:
  1. Read both markdown files
  2. Insert figure image embeds after each figure legend paragraph
  3. Add page-break between manuscript and supplement
  4. Run pandoc + xelatex to produce PDF
"""
import subprocess
import sys
import re
from pathlib import Path

PEARL_DIR  = Path("/Users/sanjaybasu/waymark-local/notebooks/pearl")
FIGS_DIR   = PEARL_DIR / "figures_png"
MS_SRC     = PEARL_DIR / "manuscript" / "PEARL_manuscript_v3.md"
SUP_SRC    = PEARL_DIR / "supplement" / "PEARL_supplement_v3.md"
OUT_PDF    = PEARL_DIR / "PEARL_combined_v3.pdf"
COMBINED   = Path("/tmp/PEARL_combined_v3.md")

# Maps legend-start text → figure path (searched in paragraph text)
MS_FIGS = {
    "**Figure 1.**": FIGS_DIR / "fig1_pipeline.png",
    "**Figure 2.**": FIGS_DIR / "fig2_drope_forest.png",
}
SUP_FIGS = {
    "**Appendix Figure 1.**": FIGS_DIR / "fig3_imi_equity.png",
    "**Appendix Figure 2.**": FIGS_DIR / "fig4_camden.png",
    "**Appendix Figure 3.**": FIGS_DIR / "fig5_sensitivity.png",
}

def inject_figures(text: str, fig_map: dict) -> str:
    """Insert a markdown image embed after each matching legend paragraph."""
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        out.append(lines[i])
        stripped = lines[i].strip()
        for legend_start, fig_path in fig_map.items():
            if stripped.startswith(legend_start) and fig_path.exists():
                # Collect the full paragraph (until blank line or next non-blank)
                while i + 1 < len(lines) and lines[i + 1].strip():
                    i += 1
                    out.append(lines[i])
                # Insert image after paragraph
                out.append("")
                out.append(f"![]({fig_path}){{width=100%}}")
                out.append("")
                break
        i += 1
    return "\n".join(out)

def build_combined():
    ms_text  = MS_SRC.read_text()
    sup_text = SUP_SRC.read_text()

    # Inject figures into each section
    ms_text  = inject_figures(ms_text,  MS_FIGS)
    sup_text = inject_figures(sup_text, SUP_FIGS)

    # Page break between manuscript and supplement
    page_break = "\n\n\\newpage\n\n---\n\n\\newpage\n\n"

    combined = ms_text + page_break + sup_text
    COMBINED.write_text(combined)
    print(f"Combined markdown written: {COMBINED}")

def run_pandoc():
    cmd = [
        "pandoc",
        str(COMBINED),
        "--from=markdown+tex_math_dollars+pipe_tables+fenced_code_blocks"
               "+raw_html+raw_tex+superscript+subscript",
        "--to=pdf",
        "--pdf-engine=xelatex",
        "--standalone",
        "--output", str(OUT_PDF),
        # Page geometry
        "-V", "geometry:margin=1in",
        # Font — Times-like via mathspec
        "-V", "mainfont=Times New Roman",
        "-V", "mathfont=Times New Roman",
        "-V", "fontsize=11pt",
        # 1.5 line spacing
        "-V", "linestretch=1.5",
        # Header includes for long tables and figure placement
        "--include-in-header=/tmp/pearl_pdf_header.tex",
        # Resource path for figures (already absolute in embed)
        f"--resource-path={PEARL_DIR}:{FIGS_DIR}",
    ]
    print("Running pandoc + xelatex...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-3000:], file=sys.stderr)
        sys.exit(1)
    print(f"PDF written: {OUT_PDF}  ({OUT_PDF.stat().st_size // 1024} KB)")

def write_latex_header():
    """Write a small LaTeX header for long-table support and figure control."""
    header = r"""\usepackage{longtable}
\usepackage{booktabs}
\usepackage{array}
\usepackage{float}
\usepackage{graphicx}
\usepackage{caption}
\floatplacement{figure}{H}
\setlength{\LTpre}{6pt}
\setlength{\LTpost}{6pt}
% Prevent figures from floating past their section
\usepackage[section]{placeins}
% Nicer table rules
\renewcommand{\arraystretch}{1.2}
"""
    Path("/tmp/pearl_pdf_header.tex").write_text(header)
    print("LaTeX header written.")

if __name__ == "__main__":
    write_latex_header()
    build_combined()
    run_pandoc()
