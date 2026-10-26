"""Build the handoff PDFs from the frozen report draft and existing figures.

Does not read prediction arrays, fit parameters, or rewrite outputs/.
"""

from __future__ import annotations

import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "report" / "draft" / "report_draft.md"
SCRIPT = Path(__file__).resolve().parent / "speaker_script.md"
HANDOFF = Path(__file__).resolve().parent
FIGURES = ROOT / "report" / "figures"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

REPORT_CSS = """
@page { size: A4; margin: 16mm 15mm 18mm; }
body { font-family: "Segoe UI", Calibri, sans-serif; font-size: 11pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 18pt; line-height: 1.25; margin-top: 0; }
h2 { font-size: 14pt; margin-top: 1.4em; page-break-after: avoid; }
h3 { font-size: 12pt; margin-top: 1.1em; page-break-after: avoid; }
p { margin: 0.45em 0; }
.banner { background: #fff4d6; border: 1px solid #e0c36a; padding: 8px 10px; margin: 0 0 14px; font-size: 10pt; }
code { font-family: Consolas, monospace; font-size: 0.92em; }
.eq { font-family: Cambria, "Times New Roman", serif; text-align: center; margin: 0.7em 1.2em; white-space: pre-wrap; }
table { border-collapse: collapse; width: 100%; margin: 0.7em 0 1em; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #b9b9b9; padding: 3px 5px; vertical-align: top; }
th { background: #f3f3f3; }
img { display: block; max-width: 100%; height: auto; margin: 0.4em auto 0.2em; page-break-inside: avoid; }
a { color: #134; }
"""

SLIDE_CSS = """
@page { size: 13.333in 7.5in; margin: 0.42in 0.5in 0.38in; }
body { font-family: "Segoe UI", Calibri, sans-serif; color: #14202b; margin: 0; }
.slide { page-break-after: always; height: 6.55in; position: relative; }
.kicker { color: #3d5c73; font-size: 13pt; letter-spacing: 0.04em; margin: 0; }
h1 { font-size: 28pt; line-height: 1.15; margin: 0.08in 0 0.16in; }
h2 { font-size: 22pt; margin: 0 0 0.12in; }
ul { font-size: 18pt; line-height: 1.35; margin: 0.08in 0 0; padding-left: 0.3in; }
li { margin: 0.06in 0; }
.note { font-size: 14pt; color: #333; }
img { display: block; max-width: 100%; max-height: 5.15in; margin: 0.08in auto 0; object-fit: contain; }
.footer { position: absolute; left: 0; bottom: 0; font-size: 11pt; color: #667; }
table { border-collapse: collapse; width: 100%; font-size: 15pt; margin-top: 0.15in; }
th, td { border-bottom: 1px solid #c5d0d8; padding: 0.06in 0.08in; text-align: right; }
th:first-child, td:first-child { text-align: left; }
"""

SCRIPT_CSS = """
@page { size: A4; margin: 16mm 16mm; }
body { font-family: "Segoe UI", Calibri, sans-serif; font-size: 11.5pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 18pt; }
h2 { font-size: 13.5pt; margin-top: 1.15em; page-break-after: avoid; }
p { margin: 0.4em 0; }
strong { font-weight: 650; }
code { font-family: Consolas, monospace; font-size: 0.92em; }
"""


def inline(text: str, base: Path) -> str:
    placeholders: list[str] = []

    def hold(fragment: str) -> str:
        placeholders.append(fragment)
        return f"\x00{len(placeholders) - 1}\x00"

    def image(match: re.Match[str]) -> str:
        alt, target = match.group(1), match.group(2)
        path = (base / target).resolve()
        src = path.as_uri() if path.exists() else html.escape(target)
        return hold(f'<img src="{src}" alt="{html.escape(alt)}">')

    def link(match: re.Match[str]) -> str:
        label, target = match.group(1), match.group(2)
        shown = inline(label, base)
        if target.lower().endswith(".png"):
            path = (base / target).resolve()
            src = path.as_uri() if path.exists() else html.escape(target)
            return hold(f'{shown}<img src="{src}" alt="{html.escape(label)}">')
        return hold(f'<a href="{html.escape(target, quote=True)}">{shown}</a>')

    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", image, text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, text)
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return re.sub(r"\x00(\d+)\x00", lambda m: placeholders[int(m.group(1))], escaped)


def markdown_to_html(text: str, base: Path) -> str:
    math: list[str] = []

    def save_math(match: re.Match[str]) -> str:
        math.append(html.escape(match.group(1).strip()))
        return f"\n\n@@MATH{len(math) - 1}@@\n\n"

    text = re.sub(r"\$\$(.+?)\$\$", save_math, text, flags=re.S)
    lines = text.splitlines()
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith("|"):
            rows: list[str] = []
            while index < len(lines) and lines[index].startswith("|"):
                rows.append(lines[index])
                index += 1
            blocks.append(render_table(rows, base))
            continue
        heading = re.match(r"^(#{1,3})\s+(.*)$", line)
        if heading:
            level = len(heading.group(1))
            blocks.append(f"<h{level}>{inline(heading.group(2), base)}</h{level}>")
            index += 1
            continue
        math_only = re.fullmatch(r"@@MATH(\d+)@@", line.strip())
        if math_only:
            blocks.append(f'<p class="eq">{math[int(math_only.group(1))]}</p>')
            index += 1
            continue
        chunks: list[tuple[str, bool]] = []
        while index < len(lines) and lines[index].strip() and not lines[index].startswith("|") and not lines[index].startswith("#"):
            raw = lines[index]
            chunks.append((raw.strip(), raw.endswith("  ")))
            index += 1
        pieces: list[str] = []
        for chunk_index, (chunk, hard_break) in enumerate(chunks):
            pieces.append(inline(chunk, base))
            if chunk_index < len(chunks) - 1:
                pieces.append("<br>" if hard_break else " ")
        body = "".join(pieces)
        body = re.sub(r"@@MATH(\d+)@@", lambda m: f'</p><p class="eq">{math[int(m.group(1))]}</p><p>', body)
        blocks.append(f"<p>{body}</p>")
    return "\n".join(blocks)


def render_table(rows: list[str], base: Path) -> str:
    parsed = [[cell.strip() for cell in row.strip().strip("|").split("|")] for row in rows]
    parsed = [row for row in parsed if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in row)]
    if not parsed:
        return ""
    head, body = parsed[0], parsed[1:]
    html_rows = ["<tr>" + "".join(f"<th>{inline(cell, base)}</th>" for cell in head) + "</tr>"]
    html_rows.extend("<tr>" + "".join(f"<td>{inline(cell, base)}</td>" for cell in row) + "</tr>" for row in body)
    return "<table>" + "".join(html_rows) + "</table>"


def document(title: str, css: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>{css}</style></head><body>{body}</body></html>"""


def slides_html() -> str:
    def fig(name: str) -> str:
        return (FIGURES / name).resolve().as_uri()

    slides = [
        ("BN5211 monocular endoscopic depth", """
        <p class="kicker">Medical Robotics Intelligence · final project</p>
        <h1>General-purpose depth on C3VD: evaluation and target-domain calibration</h1>
        <ul>
          <li>One frozen released model, not a new network</li>
          <li>Development sequence fits the baseline and two calibrations</li>
          <li>Test sequence is scored once, after the parameters are frozen</li>
        </ul>
        <p class="footer">Group [NUMBER] · names and student IDs to be inserted before submission</p>"""),
        ("Why endoscopic depth", """
        <h2>Distance from a single endoscopic image is ambiguous</h2>
        <ul>
          <li>Camera-to-surface distance supports spatial awareness in medical robotics</li>
          <li>C3VD pairs colonoscope video of a physical model with registered depth</li>
          <li>RGB is a phantom recording; ground truth is rendered, not a patient scan</li>
          <li>The question is transfer of a general model, not clinical validation</li>
        </ul>"""),
        ("Frozen protocol", """
        <h2>Development fits parameters. Test only evaluates them.</h2>
        <ul>
          <li><code>trans_t1_a</code>: 61 frames &nbsp;&nbsp; <code>trans_t2_b</code>: 103 frames</li>
          <li>Depth in mm: raw × 100 / 65535, valid where 0 &lt; raw &lt; 65535</li>
          <li>136,655,314 valid test pixels, pooled, not averaged as frames</li>
          <li>AbsRel, RMSE (mm), and the fraction with ratio strictly below 1.25</li>
        </ul>"""),
        ("Four methods", """
        <h2>Same mask, same pixels, different use of the image</h2>
        <ul>
          <li>Constant median: 17.2595 mm everywhere. No image structure</li>
          <li>Original Depth Anything V2: frozen indoor metric checkpoint</li>
          <li>Global scale: multiply by s = 0.0382144, fit on development only</li>
          <li>Log-affine: exp(a + b ln D), a = −26.3679, b = 4.7915</li>
        </ul>
        <p class="note">Only the original checkpoint is zero-shot relative to our development labels.</p>"""),
        ("Qualitative", f"""
        <h2>Test frame 0102, chosen as the last frame</h2>
        <img src="{fig("Figure_2_qualitative_0102.png")}" alt="Qualitative comparison">
        <p class="footer">Gray is the shared exclusion mask. Original predictions use 0–1300 mm; the others use 0–100 mm.</p>"""),
        ("Quantitative", f"""
        <h2>Log-affine wins AbsRel and RMSE. Global scale wins δ.</h2>
        <img src="{fig("Figure_3_quantitative_comparison.png")}" alt="Quantitative comparison">"""),
        ("Depth bins", f"""
        <h2>The δ loss is concentrated at 10–20 mm</h2>
        <img src="{fig("Figure_4_depth_dependent_error.png")}" alt="Depth-dependent error">"""),
        ("Trade-off", """
        <h2>A lower average error is not more accurate pixels</h2>
        <table>
          <tr><th>Method</th><th>AbsRel</th><th>RMSE</th><th>δ &lt; 1.25</th></tr>
          <tr><td>Constant median</td><td>0.5499</td><td>17.71 mm</td><td>0.2752</td></tr>
          <tr><td>Original DAV2</td><td>27.0687</td><td>418.09 mm</td><td>0</td></tr>
          <tr><td>Global scale</td><td>0.4685</td><td>16.69 mm</td><td>0.3092</td></tr>
          <tr><td>Log-affine</td><td>0.4191</td><td>13.12 mm</td><td>0.2897</td></tr>
        </table>
        <p class="note">Relative to global scale: AbsRel −10.54%, RMSE −21.39%, δ −1.96 percentage points.</p>"""),
        ("Limits", """
        <h2>What these numbers do not say</h2>
        <ul>
          <li>Not label-free: baseline and both calibrations use development depth</li>
          <li>Not 103 patients: one continuous test video</li>
          <li>Not recovered folds: Pearson 0.784 is large-scale association</li>
          <li>60–100 mm remains too shallow: mean GT 76 mm, log-affine about 36 mm</li>
        </ul>"""),
        ("Close", """
        <h2>Calibration changed the scale. It did not finish the geometry.</h2>
        <ul>
          <li>The released model had relative structure and a severe scale error</li>
          <li>One development scale factor beat a constant depth</li>
          <li>Log-affine improved magnitude error and paid a near-range δ penalty</li>
          <li>Report that trade-off. Do not report a new algorithm or a clinical system</li>
        </ul>
        <p class="footer">Questions</p>"""),
    ]
    parts = []
    total = len(slides)
    for index, (title, body) in enumerate(slides, start=1):
        numbered = body.replace('<p class="footer">', f'<p class="footer">{index} / {total} · ', 1)
        if '<p class="footer">' not in body:
            numbered += f'<p class="footer">{index} / {total}</p>'
        parts.append(f'<section class="slide" aria-label="{html.escape(title)}">{numbered}</section>')
    return document("BN5211 slides", SLIDE_CSS, "\n".join(parts))


def print_pdf(html_path: Path, pdf_path: Path) -> None:
    if pdf_path.exists():
        pdf_path.unlink()
    user_data = Path(tempfile.mkdtemp(prefix="bn5211-edge-"))
    command = [
        str(EDGE),
        "--headless",
        "--disable-gpu",
        "--no-first-run",
        "--no-pdf-header-footer",
        f"--user-data-dir={user_data}",
        f"--print-to-pdf={pdf_path}",
        html_path.resolve().as_uri(),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=180)
    shutil.rmtree(user_data, ignore_errors=True)
    if completed.returncode != 0 or not pdf_path.exists() or pdf_path.stat().st_size < 1000:
        raise RuntimeError(
            f"Edge failed for {pdf_path.name}: code {completed.returncode}\n"
            f"{completed.stdout[-500:]}\n{completed.stderr[-500:]}"
        )


def main() -> None:
    HANDOFF.mkdir(exist_ok=True)
    report_body = (
        '<p class="banner">Team number, names, and student IDs are still unknown and are left as placeholders. '
        "Replace them before Canvas submission. The scores below are the frozen results and were not recomputed for this PDF.</p>"
        + markdown_to_html(DRAFT.read_text(encoding="utf-8"), DRAFT.parent)
    )
    report_html = HANDOFF / "report.html"
    slides_html_path = HANDOFF / "slides.html"
    script_html = HANDOFF / "speaker_script.html"
    report_html.write_text(document("BN5211 report draft", REPORT_CSS, report_body), encoding="utf-8")
    slides_html_path.write_text(slides_html(), encoding="utf-8")
    script_html.write_text(
        document("Speaker script", SCRIPT_CSS, markdown_to_html(SCRIPT.read_text(encoding="utf-8"), SCRIPT.parent)),
        encoding="utf-8",
    )
    print_pdf(report_html, HANDOFF / "GroupNumber_report.pdf")
    print_pdf(slides_html_path, HANDOFF / "GroupNumber_slides.pdf")
    print_pdf(script_html, HANDOFF / "GroupNumber_speaker_script.pdf")
    for name in ("GroupNumber_report.pdf", "GroupNumber_slides.pdf", "GroupNumber_speaker_script.pdf"):
        path = HANDOFF / name
        print(f"{name} {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
