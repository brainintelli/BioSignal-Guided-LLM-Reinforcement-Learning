"""Reproduce the seven English figures from the bundled, verified inputs.

Usage: python figures/draw.py --all --output figures
Requires ReportLab. Use --font-regular and --font-bold to select installed
Times New Roman files on a different operating system. No network is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas


HERE = Path(__file__).resolve().parent
WIDTH = 396.0
INK = "#172B3C"
MID = "#445969"
GRID = "#C3CBD0"
FAINT = "#EDF1F3"
NAVY = "#174C73"
OCHRE = "#80521E"
WHITE = "#FFFFFF"
BLACK = "#1D252B"
REG = "FigureSerif"
BOLD = "FigureSerifBold"
FIG_FACE = "Times New Roman"
SVG = "http://www.w3.org/2000/svg"
REGISTERED_TEXT = re.compile(r"⟦([^⟧]+)⟧(.*?)⟦/⟧")
ET.register_namespace("", SVG)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def font_files(regular, bold):
    if bool(regular) != bool(bold):
        raise SystemExit("Supply both --font-regular and --font-bold.")
    if regular:
        return regular, bold
    if os.environ.get("WINDIR"):
        home = Path(os.environ["WINDIR"]) / "Fonts"
        pair = home / "times.ttf", home / "timesbd.ttf"
        if all(p.is_file() for p in pair):
            return pair
    if shutil.which("fc-match"):
        for family in ("Times New Roman", "Liberation Serif"):
            matches = []
            for weight in ("Regular", "Bold"):
                proc = subprocess.run(
                    [
                        "fc-match",
                        "-f",
                        "%{family}\n%{file}\n",
                        f"{family}:style={weight}",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                fields = proc.stdout.splitlines()
                if len(fields) == 2 and family.casefold() in fields[0].casefold():
                    matches.append(Path(fields[1]))
            if len(matches) == 2:
                return tuple(matches)
    raise SystemExit(
        "Install Times New Roman or Liberation Serif, or pass both font paths."
    )


def register_fonts(regular, bold):
    global FIG_FACE
    paths = font_files(regular, bold)
    for name, fp in zip((REG, BOLD), paths):
        face = TTFont(name, str(fp))
        family = face.face.familyName
        family = family.decode() if isinstance(family, bytes) else family
        if family not in {"Times New Roman", "Liberation Serif"}:
            raise SystemExit(
                f"Expected Times New Roman or Liberation Serif; found {family}."
            )
        if name == REG:
            FIG_FACE = family
        elif family != FIG_FACE:
            raise SystemExit("Regular and bold fonts must be from the same family.")
        pdfmetrics.registerFont(face)
    return {key: sha(fp) for key, fp in zip(("regular", "bold"), paths)}


def s(tag):
    return "{" + SVG + "}" + tag


class Artwork:
    """Paint the same physical primitives to PDF and accessible SVG."""

    def __init__(self, out, num, height, title, description):
        self.height = height
        self.stem = out / f"fig{num:02d}"
        self.pdf = Canvas(
            str(self.stem.with_suffix(".pdf")),
            pagesize=(WIDTH, height),
            initialFontName=REG,
            initialFontSize=8,
            invariant=1,
            pageCompression=1,
        )
        self.pdf.setTitle(title)
        self.pdf.setAuthor("Anonymous")
        self.pdf.setCreator("Vector figure renderer")
        self.pdf.setSubject(description)
        self.svg = ET.Element(
            s("svg"),
            {
                "width": f"{WIDTH:g}pt",
                "height": f"{height:g}pt",
                "viewBox": f"0 0 {WIDTH:g} {height:g}",
                "role": "img",
                "aria-labelledby": "chart-title chart-desc",
            },
        )
        ET.SubElement(self.svg, s("title"), {"id": "chart-title"}).text = title
        ET.SubElement(self.svg, s("desc"), {"id": "chart-desc"}).text = description
        self.target = self.svg
        self.text_boxes = []
        self.rect(0, 0, WIDTH, height, WHITE)

    def group(self, **data):
        self.target = ET.SubElement(
            self.svg,
            s("g"),
            {"data-" + k.replace("_", "-"): str(v) for k, v in data.items()},
        )

    def end(self):
        self.target = self.svg

    def text(self, x, y, value, size=8, bold=False, col=INK, align="left"):
        value = str(value)
        numeric_refs = []

        def value_text(m):
            key, precision, transform = m.group(1).rsplit("|", 2)
            numeric_refs.append(
                {
                    "number_id": key,
                    "display": m.group(2),
                    "precision": int(precision),
                    "transform": transform,
                }
            )
            return m.group(2)

        value = REGISTERED_TEXT.sub(value_text, value)
        if size < 7:
            raise ValueError("Font below the publication minimum.")
        font = BOLD if bold else REG
        self.pdf.setFont(font, size)
        self.pdf.setFillColor(HexColor(col))
        width = pdfmetrics.stringWidth(value, font, size)
        left = (
            x if align == "left" else x - width / 2 if align == "center" else x - width
        )
        if (
            left < -0.15
            or left + width > WIDTH + 0.15
            or y < 0
            or y + size * 0.23 > self.height + 0.1
        ):
            raise ValueError(
                f"Text outside {self.stem.name}: {value!r} ({left=}, {width=}, {y=})"
            )
        self.pdf.drawString(left, self.height - y, value)
        self.text_boxes.append(
            [
                round(left, 3),
                round(y - size * 0.72, 3),
                round(left + width, 3),
                round(y + size * 0.17, 3),
                value,
            ]
        )
        attr = {
            "x": str(x),
            "y": str(y),
            "font-family": f"{FIG_FACE}, Times New Roman, Times, serif",
            "font-weight": "bold" if bold else "normal",
            "font-size": str(size),
            "fill": col,
            "text-anchor": {"left": "start", "center": "middle", "right": "end"}[align],
        }
        if numeric_refs:
            attr["data-registered-values"] = json.dumps(
                numeric_refs, ensure_ascii=False, separators=(",", ":")
            )
        ET.SubElement(self.target, s("text"), attr).text = value

    def line(self, x1, y1, x2, y2, col=GRID, width=0.7, dash=None):
        self.pdf.setStrokeColor(HexColor(col))
        self.pdf.setLineWidth(width)
        self.pdf.setDash(dash or [])
        self.pdf.line(x1, self.height - y1, x2, self.height - y2)
        attr = {
            "x1": str(x1),
            "x2": str(x2),
            "y1": str(y1),
            "y2": str(y2),
            "stroke": col,
            "stroke-width": str(width),
        }
        if dash:
            attr["stroke-dasharray"] = " ".join(map(str, dash))
        ET.SubElement(self.target, s("line"), attr)
        self.pdf.setDash([])

    def rect(self, x, y, w, h, fill=WHITE, stroke=None, lw=0.65):
        self.pdf.setFillColor(HexColor(fill))
        self.pdf.setLineWidth(lw)
        if stroke:
            self.pdf.setStrokeColor(HexColor(stroke))
        self.pdf.rect(x, self.height - y - h, w, h, fill=1, stroke=int(bool(stroke)))
        ET.SubElement(
            self.target,
            s("rect"),
            {
                "x": str(x),
                "y": str(y),
                "width": str(w),
                "height": str(h),
                "fill": fill,
                "stroke": stroke or "none",
                "stroke-width": str(lw),
            },
        )

    def polygon(self, points, fill, stroke=None, lw=0.7):
        path = self.pdf.beginPath()
        x, y = points[0]
        path.moveTo(x, self.height - y)
        for x, y in points[1:]:
            path.lineTo(x, self.height - y)
        path.close()
        self.pdf.setFillColor(HexColor(fill))
        self.pdf.setLineWidth(lw)
        if stroke:
            self.pdf.setStrokeColor(HexColor(stroke))
        self.pdf.drawPath(path, fill=1, stroke=int(bool(stroke)))
        ET.SubElement(
            self.target,
            s("polygon"),
            {
                "points": " ".join(f"{x},{y}" for x, y in points),
                "fill": fill,
                "stroke": stroke or "none",
                "stroke-width": str(lw),
            },
        )

    def marker(self, x, y, kind, col=INK, r=2.7, open_=False):
        fill = WHITE if open_ else col
        if kind == "square":
            return self.rect(x - r * 0.85, y - r * 0.85, r * 1.7, r * 1.7, fill, col)
        if kind == "diamond":
            return self.polygon(
                [(x, y - r), (x + r, y), (x, y + r), (x - r, y)], fill, col
            )
        if kind == "triangle":
            return self.polygon(
                [(x, y - r), (x + r, y + r * 0.8), (x - r, y + r * 0.8)], fill, col
            )
        self.pdf.setFillColor(HexColor(fill))
        self.pdf.setStrokeColor(HexColor(col))
        self.pdf.setLineWidth(0.65)
        self.pdf.circle(x, self.height - y, r, fill=1, stroke=1)
        ET.SubElement(
            self.target,
            s("circle"),
            {
                "cx": str(x),
                "cy": str(y),
                "r": str(r),
                "fill": fill,
                "stroke": col,
                "stroke-width": ".65",
            },
        )

    def arrow(self, points, col=MID, lw=0.85):
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            self.line(x1, y1, x2, y2, col, lw)
        (x1, y1), (x2, y2) = points[-2:]
        d = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        ux, uy = (x2 - x1) / d, (y2 - y1) / d
        bx, by = x2 - 4 * ux, y2 - 4 * uy
        self.polygon(
            [(x2, y2), (bx - 2 * uy, by + 2 * ux), (bx + 2 * uy, by - 2 * ux)], col
        )

    def header(self, y, letter, title, tag=None):
        if letter:
            self.text(5, y, letter, 11, True, NAVY)
        self.text(19 if letter else 5, y, title, 10.6, True)
        if tag:
            self.text(WIDTH - 5, y, tag, 7.5, False, MID, "right")

    def ci(self, mapx, y, mu, lo, hi, mark="circle", color=NAVY):
        x1, x2 = mapx(lo), mapx(hi)
        self.line(x1, y, x2, y, color, 1.25)
        self.line(x1, y - 2.8, x1, y + 2.8, color, 0.9)
        self.line(x2, y - 2.8, x2, y + 2.8, color, 0.9)
        self.marker(mapx(mu), y, mark, color, 2.9)

    def finish(self):
        self.pdf.showPage()
        self.pdf.save()
        ET.indent(self.svg, space="  ")
        self.stem.with_suffix(".svg").write_bytes(
            ET.tostring(self.svg, encoding="utf-8", xml_declaration=True)
        )
        return self.text_boxes


class Numbers:
    def __init__(self, bundle):
        self.values = bundle["values"]
        self.figures = bundle["figures"]

    def __call__(self, key):
        return float(self.values[key].replace(",", "").replace("−", "-"))

    def fmt(self, key, precision=0, signed=False, comma=False, factor=1):
        transformation = "identity" if factor == 1 else "negate"
        if factor not in (-1, 1):
            raise ValueError("Unsupported display transformation.")
        return f"⟦{key}|{precision}|{transformation}⟧{nice(self(key) * factor, precision, signed, comma)}⟦/⟧"


def nice(x, p=0, signed=False, comma=False):
    spec = ("+" if signed else "") + ("," if comma else "") + f".{p}f"
    return format(x, spec).replace("-", "−")


def scale(x0, x1, low, high):
    return lambda value: x0 + (x1 - x0) * (value - low) / (high - low)


def axis(a, m, ticks, y, desc=None, grid_top=None, formatter=None):
    for tick in ticks:
        if grid_top is not None:
            a.line(
                m(tick),
                grid_top,
                m(tick),
                y,
                GRID,
                0.44,
                dash=None if tick == 0 else [1.4, 2.2],
            )
        a.line(m(tick), y, m(tick), y + 2, MID, 0.65)
        a.text(
            m(tick),
            y + 10,
            formatter(tick) if formatter else nice(tick),
            7.2,
            False,
            MID,
            "center",
        )
    a.line(m(ticks[0]), y, m(ticks[-1]), y, MID, 0.7)
    if desc:
        a.text((m(ticks[0]) + m(ticks[-1])) / 2, y + 21, desc, 8, False, INK, "center")


FEATURES = (
    "Log firing rate",
    "Active fraction",
    "Temporal concentration",
    "Spatial entropy",
)


def fig1(out, num):
    d = num.figures["1"]
    first = d["results"][0]
    a = Artwork(
        out,
        1,
        217,
        "Offline neural input and paired exploration",
        "Measured events from the pre-stimulation window feed an offline readout. Only the model and game exchange actions and states. Mean paired differences are shown with fixed-map confidence intervals.",
    )
    a.header(
        13, "A", "Offline replay into a game decision", f"Chip {num.fmt(first['chip'])}"
    )
    a.line(5, 19, 391, 19)
    a.text(6, 32, "Recorded events", 9, True)
    a.text(142, 32, "Standardized readout", 9, True)
    a.text(267, 32, "Policy and game", 9, True)
    start, end = map(num, d["window"])
    n_roi = num(d["roi"])
    span = end - start
    x0, x1, y0, y1 = 28, 119, 47, 97
    mx = scale(x0, x1, 0, span)
    my = scale(y1, y0, 1, n_roi)
    a.text(6, 43, "Channel rank", 7.2, False, MID)
    for tick in (0, span / 2, span):
        a.line(mx(tick), y0, mx(tick), y1, GRID, 0.45, [1.3, 1.7])
    a.line(x0, y0, x0, y1, MID)
    a.line(x0, y1, x1, y1, MID)
    a.text(x0 - 3, y0 + 3, num.fmt(d["roi"]), 7.2, False, MID, "right")
    a.text(x0 - 3, y1, "1", 7.2, False, MID, "right")
    a.group(mark="event-raster", registered_count=d["events"])
    for t, r in d["matrix"]:
        a.line(mx(t), my(r) - 1.15, mx(t), my(r) + 1.15, NAVY, 0.85)
    a.end()
    a.text(x0, y1 + 9, "0", 7.2, False, MID, "center")
    a.text(x1, y1 + 9, f"{span:g} s", 7.2, False, MID, "center")
    a.text(
        6,
        117,
        f"{num.fmt(d['events'])} events · {num.fmt(d['active'])} active channels",
        7.6,
        False,
        MID,
    )
    a.arrow([(123, 70), (137, 70)])
    for j, (label, key) in enumerate(zip(FEATURES, d["features"])):
        y = 48 + 14 * j
        a.text(142, y, label, 7.6)
        a.text(243, y, num.fmt(key, 3), 8, True, NAVY, "right")
    a.text(142, 115, "One archived window", 7.6, False, MID)
    a.arrow([(248, 70), (263, 70)])
    a.rect(267, 53, 43, 22, FAINT, GRID)
    a.text(288.5, 67, "Model", 8.4, True, NAVY, "center")
    a.rect(349, 53, 41, 22, WHITE, GRID)
    a.text(369.5, 67, "Game", 8.4, True, INK, "center")
    a.arrow([(311, 60), (347, 60)])
    a.text(329, 53, "Action", 7.2, False, MID, "center")
    a.arrow([(369, 76), (369, 95), (288, 95), (288, 77)])
    a.text(329, 89, "Next state", 7.2, False, MID, "center")
    a.text(267, 115, "Game feedback to model only", 7.6, False, MID)
    a.line(5, 125, 391, 125, GRID, 0.85)
    a.header(140, "B", "Paired exploration difference (new cells/game)")
    a.text(
        5,
        152,
        f"Paired with the all-zero group ({num.fmt(d['allzero_mean'], 2)} cells/game); "
        f"{num.fmt(first['seeds'])} seeds × {num.fmt(first['maps'])} fixed maps, {num.fmt(d['budget'])} decisions/game.",
        8,
    )
    a.text(5, 167, "Comparison group", 7.4, True, MID)
    a.text(
        283, 167, f"Mean [{num.fmt(first['confidence'])}% CI]", 7.4, True, MID, "right"
    )
    a.text(390, 167, "Positive seeds", 7.4, True, MID, "right")
    m = scale(109, 199, 0, 3)
    axis(a, m, [0, 1, 2, 3], 202, grid_top=173)
    for j, (row, label, kind, col) in enumerate(
        zip(
            d["results"],
            ("Intact group", "Unstimulated group"),
            ("circle", "diamond"),
            (NAVY, OCHRE),
        )
    ):
        y = 180 + 15 * j
        a.text(5, y + 2.5, label, 8.5)
        a.group(
            mark="paired-ci",
            contrast=row["arm"],
            mean=row["mean"],
            low=row["lo"],
            high=row["hi"],
        )
        a.ci(m, y, num(row["mean"]), num(row["lo"]), num(row["hi"]), kind, col)
        a.end()
        a.text(
            283,
            y + 2.5,
            f"{num.fmt(row['mean'], 2, True)} [{num.fmt(row['lo'], 2)}, {num.fmt(row['hi'], 2)}]",
            8.1,
            True,
            INK,
            "right",
        )
        a.text(
            390,
            y + 2.5,
            f"{num.fmt(row['positive'])}/{num.fmt(row['seeds'])}",
            8.1,
            False,
            INK,
            "right",
        )
    return a.finish()


def fig2(out, num):
    d = num.figures["2"]
    a = Artwork(
        out,
        2,
        176,
        "One recorded first-batch training decision",
        "The same neural features reach the policy and value networks. The advantage and value target are calculated over the batch.",
    )
    a.header(13, "", "One decision in the intact group")
    a.text(
        391,
        13,
        f"Training seed {num.fmt(d['training_seed'])}",
        7.5,
        False,
        MID,
        "right",
    )
    a.line(5, 19, 391, 19)
    a.text(6, 34, "Inputs on a map from the first training batch", 9.1, True)
    a.text(390, 34, f"Map {num.fmt(d['map_seed'])}", 7.7, False, MID, "right")
    a.rect(6, 41, 123, 99, FAINT)
    a.text(12, 53, "Game state", 8.6, True)
    a.text(12, 66, "Neural features", 8.6, True)
    for j, (label, key) in enumerate(zip(FEATURES, d["features"])):
        y = 80 + j * 14
        a.text(12, y, label, 7.8)
        a.text(123, y, num.fmt(key, 3), 8.1, True, NAVY, "right")
    a.line(130, 90, 139, 90, MID)
    a.line(139, 61, 139, 113, MID)
    a.arrow([(139, 61), (151, 61)])
    a.arrow([(139, 113), (151, 113)])
    a.text(155, 59, "Policy selects an action", 9.1, True)
    a.text(155, 73, f"Move one cell in {num.fmt(d['duration_game_ticks'])} tick", 8)
    a.text(155, 86, f"Game score {num.fmt(d['native_R1'], 6, True)}", 8, False, NAVY)
    a.text(155, 111, "Value network", 9.1, True)
    a.text(155, 126, f"Estimate {num.fmt(d['value'], 5)}", 8)
    a.line(252, 70, 271, 70, MID, 0.65, [1.4, 2])
    a.line(252, 122, 271, 122, MID, 0.65, [1.4, 2])
    a.text(276, 59, "Policy adapter (LoRA)", 9.1, True)
    a.text(
        276, 74, f"Batch advantage {num.fmt(d['whole_batch_advantage'], 5, True)}", 8
    )
    a.text(276, 111, "Value parameters", 9.1, True)
    a.text(276, 126, f"Target {num.fmt(d['return_target'], 5)}", 8)
    a.line(5, 146, 391, 146)
    a.text(
        6,
        158,
        "In the all-zero group, neural features are masked at both the policy and value-network inputs.",
        8,
    )
    a.text(
        6,
        170,
        f"The language-model base stays frozen. The auxiliary reward weight is β = {num.fmt(d['beta'])}.",
        8,
    )
    return a.finish()


def fig3(out, num):
    d = num.figures["3"]
    chip = num.fmt(d["chip"])
    a = Artwork(
        out,
        3,
        231,
        "Two evaluation budgets and the input conditions",
        "Recorded game actions span their actual durations. The separate stress study compares intact with all-zero and shuffled groups on matched seeds and maps.",
    )
    a.header(
        13,
        "A",
        "A fixed number of decisions",
        f"First {chip} organoid · five-group study",
    )
    a.line(5, 19, 391, 19)
    a.text(
        6,
        32,
        f"Two recorded trajectories · training seed {num.fmt(d['seed'])} · map {num.fmt(d['world'])}",
        8,
    )
    groups = (("masked", "All-zero group"), ("baseline", "Intact group"))
    longest = max(max(t["end_ticks"]) for t in d["arms"].values())
    mx = scale(88, 376, 0, longest)
    words = {"move": "Move", "navigate": "Navigate", "wait": "Wait", "gather": "Gather"}
    for j, (arm, label) in enumerate(groups):
        y = 42 + 22 * j
        track = d["arms"][arm]
        a.text(6, y + 10, label, 8.3, True)
        start = 0
        shade = FAINT if j == 0 else "#DCE8EE"
        for act, end in zip(track["actions"], track["end_ticks"]):
            x1, x2 = mx(start), mx(end)
            a.group(mark="action", arm=arm, verb=act, start_tick=start, end_tick=end)
            a.rect(x1, y, x2 - x1, 14, shade, MID, 0.52)
            a.text((x1 + x2) / 2, y + 9.8, words[act], 7.1, False, INK, "center")
            a.end()
            start = end
    axis(a, mx, list(range(0, longest + 1, 2)), 83)
    a.text(
        88, 104, "Elapsed game ticks; each segment is one decision.", 7.8, False, MID
    )
    a.line(5, 112, 391, 112)
    a.header(
        126, "B", "A fixed game-time limit", f"Second {chip} organoid · adverse starts"
    )
    a.text(
        6,
        140,
        f"{num.fmt(d['training_seeds'])} training seeds × {num.fmt(d['arm_count'])} groups",
        8.4,
        True,
    )
    a.text(143, 140, f"{num.fmt(d['frozen_models'])} frozen models", 8.4, True)
    a.text(284, 140, f"{num.fmt(d['episodes'], comma=True)} games", 8.4, True)
    a.arrow([(119, 137), (136, 137)])
    a.arrow([(244, 137), (277, 137)])
    a.text(
        6,
        153,
        f"{num.fmt(d['training_batches'])} batches × {num.fmt(d['decisions_per_batch'])} decisions",
        7.8,
    )
    a.text(
        143,
        153,
        f"{num.fmt(d['scene_count'])} starts/model × {num.fmt(d['worlds_per_scene'])} maps/start",
        7.8,
    )
    a.text(
        284, 153, f"Read at tick {num.fmt(d['horizon_ticks'])} or earlier death", 7.8
    )
    a.line(5, 162, 391, 162)
    a.text(6, 175, "Input conditions used in these comparisons", 8.6, True)
    a.text(6, 188, "First organoid", 7.8, True, MID)
    a.text(
        85,
        188,
        "Intact group: pre-stimulation recording. All-zero group: constant-zero neural input.",
        7.8,
    )
    a.text(
        85,
        199,
        "Unstimulated group: recorded at zero stimulation output; prior stimulation possible.",
        7.8,
    )
    a.text(6, 213, "Second organoid", 7.8, True, MID)
    a.text(
        85,
        213,
        "Intact group: original replay order. All-zero group: constant-zero neural input.",
        7.8,
    )
    a.text(
        85,
        224,
        f"Shuffled group: reorder {num.fmt(d['block'])}-s blocks within trial. Intact minus all-zero or shuffled.",
        7.8,
    )
    return a.finish()


def fig4(out, num):
    d = num.figures["4"]
    a = Artwork(
        out,
        4,
        176,
        "Model records and two replay checks",
        "Four batches are linked to model registration and to a frozen parameter snapshot. Policy reloading and replaying archived actions have different coverage.",
    )
    a.header(13, "A", "Linking training records to frozen parameters")
    a.line(5, 19, 391, 19)
    a.text(6, 32, "Training records", 8.8, True)
    a.text(216, 32, "Final checkpoint", 8.8, True)
    a.arrow([(132, 30), (204, 30)])
    for j, (left, right) in enumerate(
        [
            (
                "registration.json  ·  seed and input group",
                "adapter/ and critic.pt  ·  model parameters",
            ),
            (
                f"train-1…{num.fmt(d['training_batches'])}.jsonl  ·  recorded decisions",
                "optimizers.pt  ·  optimizer and RNG state",
            ),
            (
                "update.json  ·  optimization steps",
                "policy.json and manifest.json  ·  provenance",
            ),
        ]
    ):
        y = 47 + 13 * j
        a.text(6, y, left, 8.05)
        a.text(216, y, right, 8.05)
    a.line(5, 83, 391, 83)
    a.header(98, "B", "What each verification covers")
    a.text(6, 115, "Frozen-policy reload", 9.1, True, NAVY)
    a.text(208, 115, "Replay of stored actions", 9.1, True, OCHRE)
    a.line(197, 104, 197, 170)
    a.text(
        6,
        132,
        f"{num.fmt(d['reload_games'], comma=True)} games · {num.fmt(d['reload_actions'], comma=True)} decisions",
        9,
        True,
    )
    a.text(
        208,
        132,
        f"{num.fmt(d['replayed_games'], comma=True)} games · {num.fmt(d['replayed_actions'], comma=True)} actions",
        9,
        True,
    )
    a.text(
        6,
        146,
        f"{num.fmt(d['reloaded_models'])} models; first map in each scenario.",
        7.8,
    )
    a.text(
        6, 158, f"Regenerated action mismatches: {num.fmt(d['reload_mismatch'])}.", 7.8
    )
    a.text(208, 146, "Exploration, survival and restricted waiting", 7.8)
    a.text(208, 158, "matched at the cutoff. R1 score exceptions are", 7.8)
    a.text(208, 169, "reported separately in Appendix B.", 7.8)
    return a.finish()


def fig5(out, num):
    d = num.figures["5"]
    rows = d["results"]
    a = Artwork(
        out,
        5,
        235,
        "Exploration contrasts and pairwise outcomes",
        "Two short first-chip comparisons exclude zero on the fixed maps. The other three confidence intervals cross zero. The counts underneath show the many tied paired games.",
    )
    a.header(13, "A", "Mean difference in newly visible cells")
    a.line(5, 19, 391, 19)
    a.text(
        6,
        30,
        f"First three rows: {num.fmt(d['budget'])} decisions; {num.fmt(rows[0]['seeds'])} seeds × {num.fmt(rows[0]['maps'])} maps; fixed-map {num.fmt(rows[0]['confidence'])}% CIs.",
        8,
    )
    a.group(mark="shared-ci-level", levels=" ".join(r["confidence"] for r in rows[-2:]))
    a.text(
        6,
        42,
        f"Last two (low-health): tick {num.fmt(d['horizon'])} or earlier death; {num.fmt(rows[-1]['seeds'])} seeds × {num.fmt(rows[-1]['maps'])} maps; crossed {num.fmt(rows[-1]['confidence'], 3)}% CIs.",
        8,
    )
    a.end()
    a.text(6, 57, "Recording and contrast", 7.5, True, MID)
    a.text(196, 57, "Cells/game", 7.5, True, MID, "center")
    a.text(342, 57, "Mean [CI]", 7.5, True, MID, "right")
    a.text(390, 57, "Positive seeds", 7.5, True, MID, "right")
    # All five printed intervals are on one linear scale; show both sides of zero.
    mx = scale(147, 244, -3, 12)
    axis(a, mx, [-2, 0, 4, 8, 12], 147, grid_top=62)
    a.line(mx(0), 61, mx(0), 147, MID, 0.9)
    display = [
        ("First chip / first organoid", "intact − all-zero"),
        ("First chip / first organoid", "unstimulated − all-zero"),
        ("Second chip", "intact − native R1-only"),
        ("First chip / second organoid", "intact − all-zero"),
        ("First chip / second organoid", "intact − shuffled"),
    ]
    a.line(5, 110, 391, 110, GRID, 0.55)
    for j, (row, (study, groups)) in enumerate(zip(rows, display)):
        y = 70 + j * 16
        prefix = num.fmt(row["chip"]) + (" (I)" if j < 2 else " (II)" if j > 2 else "")
        a.text(6, y + 2.5, prefix, 7.8, True)
        a.text(50, y + 2.5, groups, 8.05)
        tone = OCHRE if j == 1 else NAVY if j == 0 else MID
        a.group(
            mark="paired-ci",
            contrast=row["arm"] + "-" + row["control"],
            mean=row["mean"],
            low=row["lo"],
            high=row["hi"],
        )
        a.ci(
            mx,
            y,
            num(row["mean"]),
            num(row["lo"]),
            num(row["hi"]),
            ("circle", "diamond", "square", "square", "square")[j],
            tone,
        )
        a.end()
        a.text(
            342,
            y + 2.5,
            f"{num.fmt(row['mean'], 2, True)} [{num.fmt(row['lo'], 2)}, {num.fmt(row['hi'], 2)}]",
            7.85,
            False,
            INK,
            "right",
        )
        a.text(
            383,
            y + 2.5,
            f"{num.fmt(row['positive'])}/{num.fmt(row['seeds'])}",
            8,
            False,
            INK,
            "right",
        )
    a.text(
        6,
        168,
        "I / II denote the first / second organoids on the same chip.",
        7.3,
        False,
        MID,
    )
    a.line(5, 176, 391, 176)
    a.header(190, "B", "Paired outcomes on the first organoid")
    a.text(
        390,
        190,
        f"{num.fmt(d['pairs'])} pairs per comparison",
        7.6,
        False,
        MID,
        "right",
    )
    colors = [OCHRE, "#E6EAED", NAVY]
    for j, (arm, label) in enumerate(
        [("baseline", "Intact − all-zero"), ("sham", "Unstimulated − all-zero")]
    ):
        y = 197 + 18 * j
        a.text(6, y + 8.5, label, 8.2)
        x = 101
        width = 164
        keys = d["frequency"][arm]
        for k, key in enumerate(keys):
            seg = width * num(key) / num(d["pairs"])
            a.group(mark="outcome-count", arm=arm, count=key)
            a.rect(x, y, seg, 12, colors[k], WHITE, 0.55)
            a.end()
            x += seg
        labels = (
            f"{num.fmt(keys[0])} fewer",
            f"{num.fmt(keys[1])}/{num.fmt(d['pairs'])} tied",
            f"{num.fmt(keys[2])} more",
        )
        for mark_x, col, text in zip((269, 306, 358), colors, labels):
            a.rect(mark_x, y + 3.6, 3.6, 3.6, col, MID, 0.45)
            a.text(mark_x + 5.6, y + 8.5, text, 7.25)
    return a.finish()


def fig6(out, num):
    d = num.figures["6"]
    a = Artwork(
        out,
        6,
        268,
        "Decomposing the six-decision game score",
        "Each open dot is one seed-level paired mean. Bold markers give the mean across seeds. Intervals are shown for the total and residual score differences.",
    )
    a.header(13, "", "Game-score differences after six decisions")
    a.line(5, 19, 391, 19)
    a.text(
        6,
        32,
        f"{num.fmt(d['pairs'])} paired games per contrast · {num.fmt(d['seeds'])} training seeds × {num.fmt(d['maps'])} shared maps.",
        8,
    )
    a.text(
        6,
        44,
        "Open points: seed means. Filled diamonds: overall means. Zero is shared across both panels.",
        7.8,
    )
    allpts = [
        v
        for values in d["raw_seed_differences"].values()
        for row in values
        for v in row
    ]
    allci = [
        num(k)
        for arms in d["intervals"].values()
        for bounds in arms.values()
        for k in bounds
    ]
    # Step size is graphical, chosen from the full observed range (never clips points).
    anchor = 0.005
    div = max(abs(x) for x in allpts + allci)
    stop = anchor
    while stop < div:
        stop += anchor
    mx = scale(105, 295, -stop, stop)
    for j, (arm, label, col) in enumerate(
        [
            ("baseline", "Intact − all-zero", NAVY),
            ("sham", "Unstimulated − all-zero", OCHRE),
        ]
    ):
        ybase = 62 + j * 99
        a.text(6, ybase, label, 8.8, True, col)
        a.text(
            390,
            ybase,
            f"Timing share {num.fmt(d['means'][arm]['share'], 1)}%",
            8,
            True,
            MID,
            "right",
        )
        a.line(mx(0), ybase + 3, mx(0), ybase + 80, MID, 0.75)
        for i, (component, label2) in enumerate(
            [
                ("total", "Total score"),
                ("clock", "Timing deduction"),
                ("residual", "Residual score"),
            ]
        ):
            y = ybase + 17 + i * 25
            key = d["means"][arm][component]
            sign = -1 if component == "clock" else 1
            mu = sign * num(key)
            if component == "total":
                level = d["total_level"][arm]
                label2 += f" ({num.fmt(level, 1 if arm == 'baseline' else 3)}% CI)"
            if component == "residual":
                label2 += f" ({num.fmt(d['residual_level'][arm], 1)}% CI)"
            a.text(6, y + 2.5, label2, 7.8)
            a.group(mark="seed-means", arm=arm, component=component)
            lanes = []
            for row in d["raw_seed_differences"][arm]:
                x = mx(row[i])
                lane = next(
                    (
                        j
                        for j, points in enumerate(lanes)
                        if all(abs(x - p) >= 3.8 for p in points)
                    ),
                    len(lanes),
                )
                if lane == len(lanes):
                    lanes.append([])
                lanes[lane].append(x)
                a.marker(
                    x, y - 5.5 - lane * 3.65, "circle", "#546776", 1.45, open_=True
                )
            a.end()
            if component != "clock":
                lo, hi = d["intervals"][arm][component]
                a.group(mark="score-ci", arm=arm, component=component, low=lo, high=hi)
                a.ci(mx, y, mu, num(lo), num(hi), "diamond", col)
                a.end()
            else:
                a.marker(mx(mu), y, "diamond", col, 3.05)
            a.text(
                390,
                y + 2.5,
                num.fmt(key, 6, True, factor=sign),
                8.25,
                True,
                INK,
                "right",
            )
        if j == 1:
            ticks = [-stop, 0, stop]
            axis(a, mx, ticks, ybase + 80, formatter=lambda x: nice(x, 3))
    a.text(
        6,
        263,
        "R1 points/game. Timing is shown as a deduction. Lines give the archived total and post hoc residual CIs.",
        7.6,
        False,
        MID,
    )
    return a.finish()


def fig7(out, num):
    d = num.figures["7"]
    a = Artwork(
        out,
        7,
        163,
        "Frequencies of four observed actions",
        "Frequencies use all recorded decisions in each arm. Four selected action categories are shown descriptively without group tests or interval estimates.",
    )
    a.header(13, "", "Selected actions on held-out maps")
    a.line(5, 19, 391, 19)
    a.text(
        6,
        32,
        f"Share of all {num.fmt(d['denominator'], comma=True)} executed choices in each group "
        f"({num.fmt(d['seeds'])} seeds × {num.fmt(d['maps'])} maps × {num.fmt(d['decisions'])} decisions).",
        8,
    )
    mx = scale(79, 252, 0, 50)
    axis(
        a,
        mx,
        [0, 10, 20, 30, 40, 50],
        139,
        grid_top=54,
        formatter=lambda x: nice(x) + "%",
    )
    specs = [
        ("masked", "All-zero group", "square", MID),
        ("baseline", "Intact group", "circle", NAVY),
        ("sham", "Unstimulated group", "diamond", OCHRE),
    ]
    for j, (arm, label, mark, col) in enumerate(specs):
        x = (83, 173, 252)[j]
        a.marker(x, 45, mark, col, 2.55)
        a.text(x + 5, 47.5, label, 7.7)
    a.text(390, 60, "All-zero / intact / unstimulated", 7.4, True, MID, "right")
    for j, (verb, title) in enumerate(
        [
            ("move", "Move one cell"),
            ("navigate", "Navigate"),
            ("gather", "Gather"),
            ("wait", "Wait"),
        ]
    ):
        y = 70 + 20 * j
        a.text(6, y + 2.6, title, 8.25, True)
        for i, (arm, label, mark, col) in enumerate(specs):
            count, pct = d["actions"][arm][verb]
            a.group(
                mark="action-frequency",
                arm=arm,
                action=verb,
                percent=pct,
                denominator=d["denominator"],
            )
            a.marker(mx(num(pct)), y + (i - 1) * 5.5, mark, col, 2.15)
            a.end()
        percents = " / ".join(
            num.fmt(d["actions"][arm][verb][1], 2) + "%" for arm, *_ in specs
        )
        a.text(390, y + 2.6, percents, 7.7, False, INK, "right")
    a.text(
        6,
        159,
        "The four displayed actions are a subset of all choices. Descriptive frequencies; no between-group test.",
        7.5,
        False,
        MID,
    )
    return a.finish()


PLOTS = (fig1, fig2, fig3, fig4, fig5, fig6, fig7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--all", action="store_true")
    choice.add_argument("--figure", choices=range(1, 8), type=int)
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--font-regular", type=Path)
    parser.add_argument("--font-bold", type=Path)
    parser.add_argument(
        "--boxes", type=Path, help="Optional internal proof of text bounds."
    )
    args = parser.parse_args()
    data = args.data / "figure_inputs.json"
    index = json.loads((args.data / "source_index.json").read_bytes())
    if sha(data) != index["input_file_sha256"]:
        raise SystemExit("Figure inputs differ from the recorded SHA-256.")
    register_fonts(args.font_regular, args.font_bold)
    args.output.mkdir(exist_ok=True, parents=True)
    numbers = Numbers(json.loads(data.read_bytes()))
    bounds = {}
    for n in range(1, 8) if args.all else [args.figure]:
        bounds[str(n)] = PLOTS[n - 1](args.output, numbers)
    if args.boxes:
        args.boxes.write_text(json.dumps(bounds, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
