#!/usr/bin/env python3

"""Generates all modules in the Siemens Collection"""

from __future__ import annotations

import argparse
import itertools
import math
import numbers
import os.path
import re
from pathlib import Path
from typing import Any, Iterable, TypeVar
import uuid

import lxml.builder  # type: ignore
# there *are no stubs* for lxml.builder
from lxml import etree
import numpy as np
from PySide6 import QtGui

from tqdm import tqdm

ARGS = argparse.Namespace()
PARSER = etree.XMLParser(recover=True)
T = TypeVar("T", bound=int|float)
COORD = tuple[int|float, int|float]

DEFAULT_UNIT_WIDTH = 18  # standard width of one "module" of a DIN rail-mount device
DEFAULT_UNIT_HEIGHT = 90  # standard height of a Siemens DIN rail-mount device
HEIGHT_NOSE = 45  # default height of the "nose" of the device
MARGIN = 1
SCALE = 2

COLOR_BODY = "HTMLWhiteWhiteSmoke"
COLOR_LEVER = "HTMLGraySlateGray"
COLOR_SCREW = "lightgray"
COLOR_OFF = "HTMLGreenYellowGreen"

TEXT_COLOR_TERMINAL = "gray"
TEXT_COLOR_LOGO = "#1abc9c"

FONT_SIZE_SMALL = SCALE * 1.4
FONT_SIZE_MEDIUM = SCALE * 3

LINEHEIGHT_SMALL = FONT_SIZE_SMALL
LINEHEIGHT_MEDIUM = FONT_SIZE_MEDIUM

FONT_LOGO = f"Arial Black,{FONT_SIZE_SMALL},-1,5,75,0,0,0,0,0"
FONT_TEXT = f"Arial,{FONT_SIZE_SMALL},-1,5,75,0,0,0,0,0"
FONT_TERMINAL = f"Arial Black,{FONT_SIZE_MEDIUM},-1,5,75,0,0,0,0,0"
FONT_LABEL = f"Arial,{FONT_SIZE_MEDIUM},-1,5,75,0,0,0,0,0"

DIAMETER_SCREW = 7

LINESTYLE = {
    "line-style": "normal",
    "line-weight": "normal",
    "color": "black",
    "filling": "none",
}

THINLINE = LINESTYLE | {
    "line-weight": "thin",
}

STRONGLINE = LINESTYLE | {
    "line-weight": "hight",  # sic!
}

HEAVYLINE = LINESTYLE | {
    "line-weight": "eleve",  # sic!
}

NOLINE = THINLINE | {
    "line-weight": "none",
    "color": "none",
}

ATTRIBS = {
    "antialias": "false",
}

LINEATTRIBS = ATTRIBS | {
    "end1": "none",
    "end2": "none",
    "length1": 1.5,
    "length2": 1.5,
}

RECTATTRIBS = ATTRIBS | {
    "rx": 0,
    "ry": 0,
}

FIELDATTRIBS = {
    "font": FONT_TEXT,
    "Halignment": "AlignLeft",
    "Valignment": "AlignTop",
    "frame": "false",
    "rotation": 0,
    "keep_visual_rotation": "false",
}

INFOFIELD = FIELDATTRIBS | {
    "text_from": "ElementInfo",
    "uuid": "{2a4d8931-d28f-46d3-b008-1c690a6924ec}",
}

USERFIELD = FIELDATTRIBS | {
    "text_from": "UserText",
    "uuid": "{8bb939d1-3a61-48ee-99ac-e4d3f9f6c390}",
}

LABELFIELD = INFOFIELD | {
    "font": FONT_LABEL,
}

INFORMATIONS = {
    "manufacturer": "Siemens",
    "manufacturer_reference": "",
    "machine_manufacturer_reference": "",
    "unity": "",
    "label": "",
    "quantity": "",
    "description": "",
    "comment": "",
    "plant": "",
    "supplier": "",
}


def format_decimal(num: float) -> str:
    """ Formats a number in a sensible way. """
    if int(num) == num:
        return str(int(num))
    return str(round(num, 3))


def match(d: dict, key: str, pattern: str, text: str) -> None:
    """ Helper function. """
    m = re.search(r"\b" + pattern + r"\b", text, re.UNICODE)
    if m:
        d[key] = m.group(1)


def style(s: dict[str, str], **kwargs) -> str:
    """ Helper function. """
    return ";".join([":".join(kv) for kv in (s | kwargs).items()])


def natural_sort_key_2(key):
    def f(mo):
        s = mo.group(0)
        return str(len(s)) + s

    return re.sub(r"([\d]+)", f, key)


def natural_sort_key(key):
    def f(mo):
        s = mo.group(0)
        return f"{int(s):04d}"  # 1000 mA

    return re.sub(r"([\d]+)", f, key)

def cycle(start: T, *offsets : T) -> Iterable[T]:
    """
    1/2, 0, 1 => 1/2, 1/2, 3/2, 3/2, 5/2, ...
    1,  -1, 1 => 1, 0, 1, 0, 1, ...
    """
    offset: T = start
    yield offset
    while True:
        for offs in offsets:
            offset += offs # type: ignore
            yield offset


class AffineTransform:
    """A class for affine transformations."""

    def __init__(self, width, height) -> None:
        self.at = np.array(
            [
                [1, 0, -width],
                [0, 1, -height],
                [0, 0, 1],
            ]
        ) @ np.array(
            [
                [SCALE, 0, 0],
                [0, SCALE, 0],
                [0, 0, 1],
            ]
        )

    def transform(self, x: float, y: float) -> tuple[float, float]:
        return (self.at @ [x, y, 1]).tolist()

    def scale(self, w: float, h: float) -> tuple[float, float]:
        return (self.at[0:2, 0:2] @ [w, h]).tolist()

    def tx(self, x: float) -> float:
        return self.transform(x, 0)[0]

    def ty(self, y: float) -> float:
        return self.transform(0, y)[1]

    def sx(self, x: float) -> float:
        return self.scale(x, 0)[0]

    def sy(self, y: float) -> float:
        return self.scale(0, y)[1]


class BetterElementMaker(lxml.builder.ElementMaker):
    """An ElementMaker that accepts numbers as attributes."""

    def __call__(self, tag, *children, **attrib):
        for c in children:
            if isinstance(c, dict):
                for k in list(c):
                    if isinstance(c[k], numbers.Number):
                        c[k] = format_decimal(c[k])
        for k in list(attrib):
            if isinstance(attrib[k], numbers.Number):
                attrib[k] = format_decimal(attrib[k])
        return super().__call__(tag, *children, **attrib)


E = BetterElementMaker()

DIR_NAMES: dict[str, object] = {
    "siemens": E.names(
        E.name("Siemens", lang="en"),
    ),
    "MCB": E.names(
        E.name("Jističe", lang="cs"),
        E.name("Leitungsschutzschalter", lang="de"),
        E.name("Miniature Circuit Breakers", lang="en"),
        E.name("Disjoncteurs modulaires", lang="fr"),
        E.name("Interruttori magnetotermici", lang="it"),
    ),
    "RCCB": E.names(
        E.name("Proudový chránič-jistič", lang="cs"),
        E.name("FI-Schutzschalter", lang="de"),
        E.name("Residual Current Circuit Breakers", lang="en"),
        E.name("Interrupteurs différentiels", lang="fr"),
        E.name("Interruttori differenziali", lang="it"),
    ),
    "RCBO": E.names(
        E.name("FI/LS-Schutzschalter", lang="de"),
        E.name("RCCB/MCB Combinations", lang="en"),
        E.name("Disjoncteurs différentiels combinés", lang="fr"),
        E.name("Interruttori magnetotermici differenziali", lang="it"),
    ),
    "AFDD": E.names(
        E.name("Brandschutzschalter", lang="de"),
        E.name("Arc Fault Detection Devices", lang="en"),
        E.name("Détecteurs d'arc", lang="fr"),
    ),
    "AFDD+MCB": E.names(
        E.name("Brandschutzschalter/LS-Kombis", lang="de"),
        E.name("AFDD/MCB Combinations", lang="en"),
        E.name("Disjoncteurs détecteurs d'arc combinés", lang="fr"),
    ),
    "SPD": E.names(
        E.name("Überspannungsableiter", lang="de"),
        E.name("Surge Protection Devices", lang="en"),
        E.name("Dispositifs de protection contre les surtensions", lang="fr"),
        E.name("Limitatori di sovratensione", lang="it"),
    ),
    "TERM": E.names(
        E.name("Anschlußklemmen", lang="de"),
        E.name("Terminals", lang="en"),
        E.name("Bornes", lang="fr"),
        E.name("Morsetti", lang="it"),
    ),
}


class Component:
    """Base class."""

    def __init__(self, args) -> None:
        super().__init__()
        self.data = dict(args)
        d = self.data
        for n in "width unit_width height".split():
            if n not in d:
                raise ValueError(
                    f"Missing '{n}' in component"
                )
        self.width  = d["width"]
        self.height = d["height"]
        self.unit_width = d["unit_width"]

    @staticmethod
    def read(components: list[Component], data: dict[str, Any], designation: str, description: str) -> bool:
        """Parse one line of a scraped file.

        If successful, add the device to components, and return True.
        If the device was recognized but should be skipped, return True.
        If the device was not recognized, return False.
        """
        raise NotImplementedError

    @staticmethod
    def accept(description: str) -> bool:
        """ A list of things to reject """

        if "verpack" in description:  # ignore multi-unit packages
            return False
        if "SIGRES" in description:  # hardened
            return False
        if "nur magnetische" in description:  # ignore
            return False
        if "Mobil" in description:  # FI sockets
            return False
        if "Allstrom" in description:  # AC+DC models
            return False
        if "Hilfsschalter" in description:  # with auxiliary contacts
            return False
        if "Hilfschalter" in description:  # sic! auxiliary contacts
            return False
        if "Geraeteschutzschalter1polig" in description:
            return False
        if "cULus" in description:
            return False
        return True

    def draw(self) -> lxml.etree.ElementBase:
        """Return the complete XML for the component."""
        t = AffineTransform(self.width, self.height)

        root = self.header(t)
        desc = E.description()
        root.append(desc)

        self.body(desc, t)
        return root

    def body(self, desc: lxml.etree.ElementBase, t: AffineTransform) -> None:
        """Override this and append the component body to desc."""
        raise NotImplementedError

    def get_function(self) -> str:
        """Abbreviation of the device function: MCB, RCDD, ... """
        return self.data["function"].upper()

    def get_UI_name(self) -> str | None:
        """The name that shows in the UI list in the CAD-Software"""
        return " ".join(self.get_fields("designation description"))

    def get_field(self, fields):
        """Internal helper function."""
        a = []
        for field in fields.split("|"):
            text = None
            getter = getattr(self, "get_" + field, None)
            if getter:
                text = getter()
            else:
                text = self.data.get(field, None)
            if text:
                a.append(text)
        return " ".join(a)

    def get_fields(self, fields: str | list[str]) -> list[str]:
        """Helper to build the text on the front of the device.

        The field list can be given as an array or a space-separated string.
        Each field can be a field name or a concatenation of names: In|Un
        """
        if not isinstance(fields, list):
            fields = fields.split()

        a = [self.get_field(field) for field in fields]
        return [x for x in a if x]

    @staticmethod
    def text_size(text : str, font_desc = FONT_TEXT) -> tuple [float, float, float, float]:
        """ Return the size of a given text. """
        font_name, font_size, _ = font_desc.split(",", 2)
        font = QtGui.QFont([font_name], round(float(font_size)), 400, False)
        fm = QtGui.QFontMetrics(font)
        r = fm.boundingRect(text)
        t = r.x() / SCALE, r.y() / SCALE, r.width() / SCALE, r.height() / SCALE
        # print(t)
        return t


    def draw_simple_text(self, desc, t: AffineTransform, x:float, y:float, text : str, **kwargs):
        text = str(text)
        align = kwargs.get("align", "")
        _, yy, w, h = self.text_size(text, FONT_TEXT)
        # default alignment is bottom left
        if "r" in align:
            x -= w
        if "c" in align:
            x -= w / 2
        if "t" in align:
            y += h
        if "m" in align:
            y += h / 2

        desc.append(E.text(
            text=text,
            x=t.tx(x),
            y=t.ty(y),
            font=FONT_TEXT,
        ))
        if kwargs.get("border"):
            INFLATE = 1 / SCALE
            _, yy, w, h = self.text_size(text, FONT_TEXT)
            desc.append(
                E.rect(
                    RECTATTRIBS,
                    x=t.tx(x - INFLATE),
                    y=t.ty(y + yy - INFLATE),
                    width=t.sx(w + 2 * INFLATE),
                    height=t.sy(h + 2 * INFLATE),
                    style=style(THINLINE),
                )
            )

    def header(self, t: AffineTransform) -> lxml.etree.ElementBase:
        """The file common to all components header."""

        # make it a tiny bit bigger so the body lines will be drawn in full thickness
        # see also: the affine transformation matrix above
        w = self.width + 2 * MARGIN
        h = self.height + 2 * MARGIN
        root = E.definition(
            version="0.90",
            width=round(t.sx(w), 0),
            height=round(t.sy(h), 0),
            link_type="thumbnail",
            type="element",
            hotspot_x=t.sx(w // 2),
            hotspot_y=t.sy(h // 2),
        )
        root.append(E.uuid(uuid=f"{{{uuid.uuid4()}}}"))
        name = self.get_UI_name()
        root.append(
            E.names(
                E.name(name, lang="en"),
                E.name(name, lang="de"),
            )
        )
        ei = E.elementInformations()
        for k, v in (INFORMATIONS | self.data).items():
            ei.append(E.elementInformation(str(v), name=k, show="1"))
        root.append(ei)
        root.append(
            E.informations(
                "Public Domain. No warranty for the accuracy of the content."
            )
        )
        return root

    def screw(
        self, desc, t: AffineTransform, cx: float, cy: float, diameter: float
    ) -> None:
        """ Draw a screw. """

        # E.circle needs the top left, not the center
        r = diameter / 2
        sin45 = math.sin(math.radians(45))
        r2 = sin45 * r * 0.82  # long groove
        r3 = sin45 * r * 0.5  # short groove
        r4 = r * 0.5  # small cross
        desc.extend(
            [
                E.circle(
                    ATTRIBS,
                    x=t.tx(cx - r),
                    y=t.ty(cy - r),
                    diameter=t.sx(diameter),
                    style=style(THINLINE, filling=COLOR_SCREW),
                ),
                E.line(
                    LINEATTRIBS,
                    x1=t.tx(cx - r2),
                    y1=t.ty(cy - r2),
                    x2=t.tx(cx + r2),
                    y2=t.ty(cy + r2),
                    style=style(STRONGLINE),
                ),
                E.line(
                    LINEATTRIBS,
                    x1=t.tx(cx - r3),
                    y1=t.ty(cy + r3),
                    x2=t.tx(cx + r3),
                    y2=t.ty(cy - r3),
                    style=style(STRONGLINE),
                ),
                E.line(
                    LINEATTRIBS,
                    x1=t.tx(cx - r4),
                    y1=t.ty(cy),
                    x2=t.tx(cx + r4),
                    y2=t.ty(cy),
                    style=style(THINLINE),
                ),
                E.line(
                    LINEATTRIBS,
                    x1=t.tx(cx),
                    y1=t.ty(cy - r4),
                    x2=t.tx(cx),
                    y2=t.ty(cy + r4),
                    style=style(THINLINE),
                ),
            ]
        )

    def draw_terminals(self, desc, t: AffineTransform) -> None:
        """Draw the component's terminals.

        Draw the terminals in a top-down then left-right order. A terminal name of "-"
        draws nothing.
        """

        d = self.data
        terminals: list[str]
        if isinstance(d["terminals"], list):
            terminals = d["terminals"]
        else:
            terminals = d["terminals"].split()

        if d.get("double_density") or d.get("fat"):
            xoffsets = [1/4, 0,  1/2]
            yoffsets = [0,   1, -1]
        else:
            xoffsets = [1/2, 0, 1]
            yoffsets = [0,   1, -1]

        for name, x, y in zip(terminals, cycle(*xoffsets), cycle(*yoffsets)):
            if name == "-":
                continue
            top = y < 0.5
            x *= self.unit_width
            y *= self.height
            desc.append(
                E.terminal(
                    x=t.tx(x),
                    y=t.ty(y),
                    type="Generic",
                    name=name,
                    uuid=f"{{{uuid.uuid4()}}}",
                    orientation="n" if top else "s",
                )
            )

            offs = d.get("terminal_offset") or min(self.width, DEFAULT_UNIT_WIDTH) / 2
            cx = x
            cy = offs if top else self.height - offs

            diameter_screw = min(DIAMETER_SCREW, self.unit_width * 0.8)
            radius = diameter_screw / 2
            self.screw(desc, t, cx, cy, diameter_screw)

            # the terminal names
            y = cy - diameter_screw / 2
            _, _, w, h = self.text_size(name, FONT_TERMINAL)
            desc.append(
                E.text(
                    text=name,
                    x=t.tx(
                        cx - w / 2
                    ),  # AAARGH!! there is no centering for text
                    y=t.ty(cy + radius + 2 + h if top else cy - radius - 3),
                    HAlignment="AlignCenter",
                    color=TEXT_COLOR_TERMINAL,
                    font=FONT_TERMINAL,
                    rotation=0,
                )
            )


class Terminal(Component):
    """Base class for all terminals."""

    def __init__(self, args) -> None:
        super().__init__(args)
        d = self.data
        d["function"] = "term"

    def get_section(self) -> str | None:
        if "section" in self.data:
            return f"{self.data["section"]}²"
        return None

    def body(self, desc: lxml.etree.ElementBase, t: AffineTransform) -> None:
        self.casing(desc, t)
        self.draw_terminals(desc, t)
        self.texts(desc, t)

    def casing(self, desc, t: AffineTransform) -> None:
        color = self.data["color"]
        stripe = None
        if color == "yellow-green":
            color = "yellow"
            stripe = "green"

        desc.append(
            E.rect(
                RECTATTRIBS,
                x=t.tx(0),
                y=t.ty(0),
                width=t.sx(self.width),
                height=t.sy(self.height),
                style=style(LINESTYLE),
            )
        )
        # make the fill the same height as the stripe, overfills half of the line width
        # of the outline
        desc.append(
            E.rect(
                RECTATTRIBS,
                x=t.tx(0),
                y=t.ty(0),
                width=t.sx(self.width),
                height=t.sy(self.height),
                style=style(NOLINE, filling=color),
            )
        )
        if stripe:
            desc.append(
                E.rect(
                    RECTATTRIBS,
                    x=t.tx(self.width * 0.5),
                    y=t.ty(0),
                    width=t.sx(self.width * 0.5),
                    height=t.sy(self.height),
                    style=style(NOLINE, filling=stripe),
                )
            )

    def texts(self, desc, t: AffineTransform) -> None:
        section = self.get_section()
        if section:
            self.draw_simple_text(desc, t, self.width * 0.5, self.height * 0.5, section, align="cm")


    COLORS = {
        "GRAU"      : "lightgray",
        "BLAU"      : "HTMLBlueDodgerBlue",
        "ROT"       : "red",
        "GRÜN"      : "green",
        "ORANGE"    : "orange",
        "GELB"      : "yellow",
        "GRÜN-GELB" : "yellow-green",
        "SCHWARZ"   : "black",
    }

    @staticmethod
    def read(components: list[Component], data: dict[str, Any], designation: str, description: str) -> bool:
        if not designation.startswith("8WH"):
            # somebody else's problem
            return False
        if not designation.startswith("8WH1000-0"):
            # our problem but we handle simple terminals only
            return True

        match(data, "section", r"([,\d]+) ?MM²", description)   # 2,5mm²

        DIMEN = {
            "F" : ( 5,   48),
            "G" : ( 6,   48),
            "H" : ( 8,   48),
            "J" : (10,   48),
            "K" : (12,   55.5),
            "M" : (16,   63.2),
            "N" : (20,   75.45),
            "Q" : (25,   88.57),
            "S" : (15.5, 52.6),
            "U" : (18,   52.6),
        }
        dimen = DIMEN[designation[10]]

        data["units"] = 1
        data["unit_width"] = dimen[0]
        data["height"] = dimen[1]
        data["width"] = data["units"] * data["unit_width"]
        data["poles"] = "1P"
        data["terminals"] = "1 2".split()
        data["terminal_offset"] = 10

        for color in Terminal.COLORS.keys():
            if color in description:
                data["color"] = Terminal.COLORS[color]

        components.append(Terminal(data))
        return True


class Device(Component):
    """Base class for all devices."""

    def __init__(self, args) -> None:
        super().__init__(args)
        self.text_top: float = 0
        self.text_bottom: float = 0
        d = self.data
        for n in "units".split():
            if n not in d:
                raise ValueError(
                    f"Missing '{n}' in device specification: {d["description"]}"
                )

    def body(self, desc: lxml.etree.ElementBase, t: AffineTransform) -> None:
        self.casing(desc, t)
        self.draw_terminals(desc, t)
        self.controls(desc, t)
        self.texts(desc, t)
        self.label(desc, t)

    # getter functions
    def get_curve(self) -> str | None:
        if "curve" in self.data:
            return f"Curve {self.data["curve"]}"
        return None

    def get_In(self) -> str | None:
        if "In" in self.data:
            return f"{format_decimal(self.data['In'])} A"
        return None

    def get_Inc(self) -> str | None:
        if "Inc" in self.data:
            return f"{format_decimal(self.data["Inc"])} kA"
        return None

    def get_Idn(self) -> str | None:
        if "Idn" in self.data:
            return f"{self.data['Idn']} mA"
        return None

    def get_Un(self) -> str:
        if "Un" in self.data:
            return f"{self.data['Un']} V"
        return ""

    def get_rc_type(self) -> str | None:
        if "rc_type" in self.data:
            return f"Type {self.data["rc_type"]}"
        return None

    # composite getter functions
    def get_curveIn(self) -> str | None:
        d = self.data
        if "curve" in d and "In" in d:
            return f"{d['curve']}{format_decimal(d['In'])}"
        return None

    def get_more(self) -> str:
        """Get more relevant attributes."""
        d = self.data
        a = []
        if "delayed" in d:
            a.append("Delayed")
        if "selective" in d:
            # selective is always delayed
            a.append("Selective")
        if d.get("double_density"):
            a.append("Double Density")
        if "out_screwless" in d:
            a.append("Screwless outputs")
        if "left_n" in d:
            a.append("Left N")
        if "pigtail" in d:
            a.append("Pigtail")
        if len(a):
            return " ".join(a)
        return ""

    @staticmethod
    def parse_poles(poles: str) -> tuple[int, int]:
        # Calculate the number of units given the poles spec
        # 1    -> 1, 0
        # 2P   -> 2, 0
        # 3+N  -> 3, 1
        # 3P+N -> 3, 1
        m = re.match(r"(\d)P?([+]N)?", poles)
        if m:
            p = int(m.group(1))
            n = 1 if m.group(2) else 0
            return p, n
        raise ValueError(f"Bogus format for poles: ({poles})")


    def casing(self, desc, t: AffineTransform) -> None:
        # outline around body, filled gray
        desc.append(
            E.rect(
                RECTATTRIBS,
                x=t.tx(0),
                y=t.ty(0),
                width=t.sx(self.width),
                height=t.sy(self.height),
                style=style(LINESTYLE, filling=COLOR_BODY),
            )
        )
        # horizontal lines
        off = (self.height - HEIGHT_NOSE) / 2
        for y in [off, self.height - off]:
            desc.append(
                E.line(
                    LINEATTRIBS,
                    x1=t.tx(0),
                    x2=t.tx(self.width),
                    y1=t.ty(y),
                    y2=t.ty(y),
                    style=style(LINESTYLE),
                )
            )
        self.text_top = off
        self.text_bottom = self.height - off

    def test_button(self, desc, t: AffineTransform, unit: int) -> None:
        BWIDTH = 13
        BHEIGHT = 6
        BY = 0
        off = (self.height - HEIGHT_NOSE) / 2
        desc.append(
            E.rect(
                RECTATTRIBS,
                x=t.tx(unit * self.unit_width + (self.unit_width - BWIDTH) / 2),
                y=t.ty(off + BY),
                width=t.sx(BWIDTH),
                height=t.sy(BHEIGHT),
                style=style(THINLINE, filling=COLOR_LEVER),
            )
        )
        if unit == 0:
            self.text_top += BHEIGHT

    def levers(self, desc, t: AffineTransform, first_unit: int, last_unit: int) -> None:
        LEVER_WIDTH = 8
        LEVER_HEIGHT = 18
        GREEN_HEIGHT = 4
        BAR_HEIGHT = 5
        LEVER_MARGIN_X = 5
        BAR_MARGIN_X = 1

        offs = (self.height - HEIGHT_NOSE) / 2
        lever_top = self.height - offs - LEVER_HEIGHT
        bar_top = self.height - offs - BAR_HEIGHT

        for u in range(first_unit, last_unit + 1):
            lever_left = LEVER_MARGIN_X + u * self.unit_width
            # black lever rect
            desc.append(
                E.rect(
                    RECTATTRIBS,
                    x=t.tx(lever_left),
                    y=t.ty(lever_top),
                    width=t.sx(LEVER_WIDTH),
                    height=t.sy(LEVER_HEIGHT),
                    style=style(THINLINE, filling=COLOR_LEVER),
                )
            )
            # green lever rect
            desc.append(
                E.rect(
                    RECTATTRIBS,
                    x=t.tx(lever_left),
                    y=t.ty(lever_top),
                    width=t.sx(LEVER_WIDTH),
                    height=t.sy(GREEN_HEIGHT),
                    style=style(THINLINE, filling=COLOR_OFF),
                )
            )
        # lever bar
        x = self.unit_width * first_unit
        width = self.width - x
        desc.append(
            E.rect(
                RECTATTRIBS,
                x=t.tx(x + BAR_MARGIN_X),
                y=t.ty(bar_top),
                width=t.sx(width - 2 * BAR_MARGIN_X),
                height=t.sy(BAR_HEIGHT),
                style=style(THINLINE, filling=COLOR_LEVER),
            )
        )

        # the thin line above the levers
        desc.append(
            E.line(
                LINEATTRIBS,
                x1=t.tx(x),
                x2=t.tx(x + width),
                y1=t.ty(lever_top),
                y2=t.ty(lever_top),
                style=style(THINLINE),
            )
        )
        self.text_bottom = lever_top

    def draw_trip_indicators(self, desc, t: AffineTransform, offsets: Iterable[COORD]) -> None:
        TIWIDTH = 10
        TIHEIGHT = 4
        for x, y in offsets:
            desc.append(
                E.rect(
                    RECTATTRIBS,
                    x=t.tx(self.unit_width * x - TIWIDTH / 2),
                    y=t.ty(self.height * y),
                    width=t.sx(TIWIDTH),
                    height=t.sy(TIHEIGHT),
                    style=style(THINLINE, filling=COLOR_OFF),
                )
            )

    def controls(self, desc, t: AffineTransform) -> None:
        raise NotImplementedError("Not Implemented")

    def texts(self, desc, t: AffineTransform) -> None:
        # manufacturer name
        self.text_top += LINEHEIGHT_SMALL
        desc.append(
            E.text(
                text="SIEMENS",
                x=t.tx(1),
                y=t.ty(self.text_top),
                color=TEXT_COLOR_LOGO,
                font=FONT_TEXT,
            )
        )
        if self.data["units"] > 1:
            self.text_top += LINEHEIGHT_SMALL * 0.3
        self.text(desc, t, self.get_fields("designation"))

    def text(self, desc, t: AffineTransform, texts: list[str]) -> None:
        for z, text in enumerate(texts):
            self.text_top += LINEHEIGHT_SMALL
            self.draw_simple_text(desc, t, 1, self.text_top, text)

    def label(self, desc, t: AffineTransform) -> None:
        desc.append(
            E.dynamic_text(
                E.text("-F1"),
                E.info_name("label"),
                LABELFIELD,
                x=t.tx(0),
                # Alignment is completely broken !!!
                # so fudge it with LINEHEIGHT
                y=t.ty(self.text_bottom - 1.5 * LINEHEIGHT_MEDIUM),
                text_width=t.sx(self.width),
            )
        )

    @staticmethod
    def _read(d: dict[str, Any], designation: str, description: str) -> None:
        """Parse one line of a scraped file and produce a device or None."""

        t = description
        t = t.replace("-POLIG", "P").replace("POLIG", "P")

        # fmt: off
        match(d, "curve",   r"([BCD])- ?CHAR", t)
        match(d, "In",      r"([.,\d]+) ?A", t)   # 25A 1,6A  1.6A
        match(d, "Inc",     r"([.,\d]+) ?KA", t)  # 6kA 4.5kA 4,5kA
        match(d, "Un",      r"([-/\d]+) ?V", t)   # 230V  230-400 V 230/400V
        match(d, "rc_type", r"TYP ([ABCF+]+)", t)
        match(d, "Idn",     r"(\d+) ?MA", t)
        match(d, "units",   r"(\d)TE", t)
        # fmt: on

        # 1P 3P+N 1+N 4+NP but not: 1
        m = re.search(r"\b(\d)(P?[+]NP?|P)\b", t)
        if m:
            d["poles"] = m.group(0)

        # C25 C25A C 25A
        m = re.search(r"\b([ABCDE]|F1|F2) ?([\d,.]+)A?\b", t)
        if m:
            d["curve"] = m.group(1)
            d["In"] = m.group(2)

        if "In" not in d or "curve" not in d:
            # C, 25A
            m = re.search(r"\b([ABCDE]|F1|F2), ([\d,.]+)A\b", t)
            if m:
                d["curve"] = m.group(1)
                d["In"] = m.group(2)

        # 1-16A 230V
        m = re.search(r"\b\d+-(\d+)A (\d+)V\b", t)
        if m:
            d["In"] = m.group(1)
            d["Un"] = m.group(2)

        # DC 24V
        m = re.search(r"\bDC ([\d]+)V\b", t)
        if m:
            d["DC"] = True
            d["Un"] = m.group(1)

        # 230/400V
        m = re.search(r"\b\d+/(\d+)V\b", t)
        if m:
            d["Un"] = m.group(1)

        if "KURZZEITVERZÖGERT" in t:
            d["delayed"] = True
        if "SELEKTIV" in t:
            d["selective"] = True
        if "N-LINKS" in t or "N LINKS" in t:
            d["left_n"] = True
        if "KOMPAKT" in t.lower():
            d["double_density"] = True
        if "PIGTAIL" in t:
            d["pigtail"] = True

        if "poles" in d:
            poles, n = Device.parse_poles(d["poles"])
            d["poles"] = f"{poles}P+N" if n == 1 else f"{poles}P"

            if "units" not in d:
                d["units"] = poles + n
                if d.get("double_density"):
                    d["units"] //= 2
            else:
                d["units"] = int(d["units"])

            if "terminals" not in d:
                terminals = [str(i) for i in range(1, 2 * poles + 1)]
                if n > 0:
                    nn = ["N", "N"]
                    if "left_n" in d:
                        terminals = nn + terminals
                    else:
                        terminals = terminals + nn
                d["terminals"] = terminals

        for field in ["In", "Inc"]:
            if field in d:
                d[field] = float(d[field].replace(",", "."))
        for field in ["Idn"]:
            if field in d:
                d[field] = int(d[field])

        d.setdefault("unit_width", DEFAULT_UNIT_WIDTH)
        d.setdefault("width", d["unit_width"] * d["units"])
        d.setdefault("height", DEFAULT_UNIT_HEIGHT)


class MCB(Device):
    """A miniature current breaker."""

    def __init__(self, args):
        super().__init__(args)
        d = self.data
        d["function"] = "mcb"
        d["label"] = "-F1"
        for n in "In curve Inc".split():
            if n not in d:
                raise ValueError(
                    f"Missing '{n}' in MCB specification: {d["description"]}"
                )

    @staticmethod
    def read(components: list[Component], data: dict[str, Any], designation: str, description: str) -> bool:

        if not designation.startswith("5S"):
            return False

        if "LEITUNGSSCHUTZSCHALTER" not in description:
            return False

        if designation[2] in ("J", "Y"):
            # 70mm american sized
            return True

        if re.match("5SP4...-[678]", designation):
            # 70mm american sized unmetered zone
            return True

        if "HAUPTLEITUNGSSCHUTZSCHALTER" in description:
            data["Inc"] = "25"

        Device._read(data, designation, description)
        components.append(MCB(data))
        return True

    def controls(self, desc, t: AffineTransform) -> None:
        units = self.data["units"]
        self.levers(desc, t, 0, units - 1)

    def texts(self, desc, t: AffineTransform) -> None:
        super().texts(desc, t)
        self.text(desc, t, self.get_fields("function curveIn Inc Un"))


class RCCB(Device):
    """A pure residual current protection device."""

    def __init__(self, args):
        super().__init__(args)
        d = self.data
        d["function"] = "rccb"
        d["label"] = "-F1"
        for n in "In rc_type Idn".split():
            if n not in d:
                raise ValueError(
                    f"Missing '{n}' in MCB specification: {d["description"]}"
                )

    @staticmethod
    def read(components: list[Component], data: dict[str, Any], designation: str, description: str) -> bool:
        if "FI-SCHUTZSCHALTER" in description or "FI-BLOCK" in description:
            Device._read(data, designation, description)
            components.append(RCCB(data))
            return True
        return False

    def controls(self, desc, t: AffineTransform) -> None:
        units = self.data["units"]
        self.levers(desc, t, units - 1, units - 1)
        self.test_button(desc, t, units - 1)

    def texts(self, desc, t: AffineTransform) -> None:
        super().texts(desc, t)
        self.text(desc, t, self.get_fields("function|rc_type Idn In Un"))
        self.draw_selective(desc, t)

    def draw_selective(self, desc, t: AffineTransform) -> None:
        if self.data.get("selective"):
            self.text_top += 2 * LINEHEIGHT_SMALL
            self.draw_simple_text(
                desc,
                t,
                self.unit_width * 0.5,
                self.text_top,
                "S",
                align="cm",
                border=True
            )


class AFDD(Device):
    def __init__(self, args):
        super().__init__(args)
        d = self.data
        d["function"] = "afdd"
        d["label"] = "-F1"
        for n in "In".split():
            if n not in d:
                raise ValueError(
                    f"Missing '{n}' in MCB specification: {d["description"]}"
                )

    @staticmethod
    def read(components: list[Component], data: dict[str, Any], designation: str, description: str) -> bool:
        if "BRANDSCHUTZSCHALTER-BLOCK" in description:
            Device._read(data, designation, description)
            components.append(AFDD(data))
            return True
        return False

    def controls(self, desc, t: AffineTransform) -> None:
        units = self.data["units"]
        self.test_button(desc, t, units - 1)

    def texts(self, desc, t: AffineTransform) -> None:
        super().texts(desc, t)
        self.text(desc, t, self.get_fields("function In Un"))


class AFDB(MCB, AFDD):
    def __init__(self, args):
        super().__init__(args)
        d = self.data
        d["function"] = "afdd+mcb"
        d["label"] = "-F1"

    @staticmethod
    def read(components: list[Component], data: dict[str, Any], designation: str, description: str) -> bool:
        if "BRANDSCHUTZSCHALTER-LS-KOMBI" in description:
            Device._read(data, designation, description)
            components.append(AFDB(data))
            return True
        return False

    def controls(self, desc, t: AffineTransform) -> None:
        units = self.data["units"]
        self.levers(desc, t, 0, units - 1)

    def texts(self, desc, t: AffineTransform) -> None:
        Device.texts(self, desc, t)  # avoid duplication
        self.text(desc, t, self.get_fields("function curveIn Inc Un"))


class RCBO(MCB, RCCB):
    """A residual current device with overload protection."""

    def __init__(self, args):
        super().__init__(args)
        d = self.data
        d["function"] = "rcbo"
        d["label"] = "-F1"

    @staticmethod
    def read(components: list[Component], data: dict[str, Any], designation: str, description: str) -> bool:
        if "FI/LS" in description:
            Device._read(data, designation, description)
            components.append(RCBO(data))
            return True
        return False

    def controls(self, desc, t: AffineTransform) -> None:
        units = self.data["units"]
        self.levers(desc, t, 0, units - 1)
        self.test_button(desc, t, units - 1)

    def texts(self, desc, t: AffineTransform) -> None:
        Device.texts(self, desc, t)  # avoid duplication
        self.text(desc, t, self.get_fields("function|rc_type curveIn|Idn Inc Un"))
        self.draw_selective(desc, t)


class SPD(Device):
    def __init__(self, args):
        super().__init__(args)
        d = self.data
        d["function"] = "spd"
        d["label"] = "-F1"
        for n in "connection spd_type".split():
            if n not in d:
                raise ValueError(
                    f"Missing '{n}' in SPD specification: {d["description"]}"
                )

    def get_spd_type(self) -> str | None:
        if "spd_type" in self.data:
            return "|".join([f"T{t}" for t in self.data["spd_type"]])
        return None


    @staticmethod
    def parse_connection(connection: str) -> tuple[int, int]:
        # Calculate the number of units given the connection spec
        # 1+0  -> 1, 0
        # 1+1  -> 1, 1
        # 3+0  -> 3, 0
        # 3+1  -> 3, 1
        # N+L  -> 1, 0
        m = re.match(r"(\d)[+](\d)", connection)
        if m:
            p = int(m.group(1))
            n = int(m.group(2))
            return p, n
        m = re.match(r"([LNPE]+)-([LNPE]+)", connection)
        if m:
            p = 1
            n = 0
            return p, n
        raise ValueError(f"Bogus format for connection: ({connection})")


    @staticmethod
    def fat_terminals(terminals: list[str]) -> list[str]:
        r : list[str] = []
        for pair in itertools.batched(terminals, 2):
            r.extend(pair)
            r.extend(pair)
        return r

    @staticmethod
    def read(components: list[Component], data: dict[str, Any], designation: str, description: str) -> bool:
        #--------2--  1+1
        #--------3--  3+0
        #--------4--  3+1

        #----------1  Typ 1 fett
        #----------2  Typ 1+2
        #----------3  Typ 1+2 fern

        # "5SD7412-1" Typ 1 units = 4
        # "5SD7412-2" Typ 1+2
        # "5SD7413-1" Typ 1 units = 6
        # "5SD7413-2" Typ 1+2
        # "5SD7413-3" Typ 1+2 fern
        # "5SD7414-1" Typ 1 units = 8
        # "5SD7414-2" Typ 1+2
        # "5SD7414-3" Typ 1+2 fern

        # "5SD7442-1" Typ 1+2 1+1 4 units
        # "5SD7443-1" Typ 1+2 3+0 6 units
        # "5SD7444-1" Typ 1+2 3+1 8 units

        # "5SD7422-0" schmale Bauform
        # "5SD7422-1" schmale Bauform + fern
        # "5SD7422-2" 49
        # "5SD7422-3" 49 + fern

        # "5SD7432-5" Typ 3  24V 18mm
        # "5SD7432-6" Typ 3 120V 18mm
        # "5SD7432-7" Typ 3 240V 18mm

        if not designation.startswith("5SD"):
            return False

        if "2+V SCHALTUNG" in description:
            # ignore DC
            return True
        if designation == "5SD7411-2":
            # not modular
            return True
        if designation == "5SD7441-1KF00":
            # not modular
            return True
        if re.match(r"5SD7432-[567]", designation):
            # cannot draw these kind of terminals yet
            return True

        description = description.replace("V A.C.", "V AC")
        description = description.replace("TYP 1,12,5KA", "TYP 1, 12,5KA")
        SPD._read(data, designation, description)
        components.append(SPD(data))
        return True

    @staticmethod
    def _read(data: dict[str, Any], designation: str, description: str) -> None:

        # ad-hoc fixes to the catalog data
        if designation == "5SD7441-1KF00":
            data["spd_type"] = ["1", "2"]

        if "spd_type" not in data:
            m = re.search(r"\bTYP (\d)[ +]*(\d)?\b", description) # TYP 1 + 2
            if m:
                data["spd_type"] = list(filter(None, m.groups()))
        if "spd_type" not in data:
            m = re.search(r"\bT([123])(?:/T([23]))?\b", description)
            if m:
                data["spd_type"] = list(filter(None, m.groups()))

        conn = None
        m = re.search(r"([-+\dLNPE]+) SCHALTUNG", description)
        if m:
            conn = m.group(1)
            if conn == "L-N":
                data["terminals"] = ["L", "N"]
            if conn == "N-PE":
                data["terminals"] = ["PEN", "N"]

        if not conn:
            # "Schaltung" missing in description
            if designation == "5SD7412-2":
                conn = "1+1"
                data["terminals"] = ["PEN", "L", "PEN", "N"]
            if designation == "5SD7481-1":
                conn = "1+0"
                data["terminals"] = [""] * 4  # impossible to say
            if designation == "5SD7483-5":
                conn = "3+0"
                data["terminals"] = ["-", "L1", "PEN", "L2", "PEN", "L3"]

        if not conn:
            raise ValueError("Cannot guess SPD connection")

        p, n = SPD.parse_connection(conn)
        data["units"] = p + n

        m = re.match("5SD7413-[23]", designation) or re.match("5SD7463-[01]", designation)
        if m:
            data["terminals"] = ["-", "L1", "PEN", "L2", "PEN", "L3"]

        m = re.match("5SD7414-[23]", designation) or re.match("5SD7464-[01]", designation)
        if m:
            data["terminals"] = ["-", "L1", "-", "L2", "⏚", "L3", "⏚", "N"]

        m = re.match("5SD7473-1", designation)
        if m:
            data["terminals"] = ["-", "L1", "⏚", "L2", "⏚", "L3"]


        # fat devices have two of each terminal
        # https://www.automation.siemens.com/bilddb/interfaces/InterfaceImageDB.asmx/GetImageVariant?objectkey=P_I201_XX_05218&amp;imagevariantid=16&amp;lang=&amp;interfaceuserid=MALL
        if "terminals" not in data:
            terminals = ["-"] * (2 * (p + n))
            if n == 0:
                terminals[0] = "PEN"
            else:
                terminals[0] = "N"
                terminals[1] = "⏚"
            for i in range(0, p):
                terminals[2*(i+n)+1] = f"L{i+1}"

            data["terminals"] = terminals

        # "fat" SPDs
        m = re.match("5SD74[14].-1", designation)
        if m:
            data["unit_width"] = 2 * DEFAULT_UNIT_WIDTH
            data["fat"] = True
            data["terminals"] = SPD.fat_terminals(data["terminals"])

        data.setdefault("connection", conn)
        data.setdefault("unit_width", DEFAULT_UNIT_WIDTH)
        data.setdefault("width", data["unit_width"] * data["units"])

        Device._read(data, designation, description)
        data.pop("poles", None)

    def draw_spd_types(self, desc, t: AffineTransform, offsets: Iterable[COORD]):
        spd_types = self.data["spd_type"]
        if len(spd_types) == 2:
            offsets2 = list(zip(spd_types, [-0.15, 0.15], [0.0, 0.0]))
        else:
            offsets2 = list(zip(spd_types, [0.0], [0.0]))

        for x, y in offsets:
            for spd_type, dx, dy in offsets2:
                self.draw_simple_text(
                    desc,
                    t,
                    self.unit_width * (x + dx),
                    self.height * (y + dy),
                    f"T{spd_type}",
                    align="cm",
                    border=True
                )

    def casing(self, desc, t: AffineTransform) -> None:
        super().casing(desc, t)

        # vertical lines
        off = (self.height - HEIGHT_NOSE) / 2
        for u in range(1, self.data["units"]):
            desc.append(
                E.line(
                    LINEATTRIBS,
                    x1=t.tx(u * self.unit_width),
                    x2=t.tx(u * self.unit_width),
                    y1=t.ty(off),
                    y2=t.ty(self.height - off),
                    style=style(LINESTYLE),
                )
            )

    def controls(self, desc, t: AffineTransform) -> None:
        units = self.data["units"]
        xoffsets = [0.5, 1]

        yoffsets = [0.5] * units
        self.draw_trip_indicators(desc, t, zip(cycle(*xoffsets), yoffsets))

        yoffsets = [0.65] * units
        self.draw_spd_types(desc, t, zip(cycle(*xoffsets), yoffsets))


    def texts(self, desc, t: AffineTransform) -> None:
        super().texts(desc, t)


def directory(directory: str, subdirs: list[str]) -> str:
    """Make all the subdirs and eventually put a description file into them."""

    Path(directory).mkdir(parents=True, exist_ok=True)
    for subdir in subdirs:
        directory = os.path.join(directory, natural_sort_key(subdir))
        try:
            Path(directory).mkdir(parents=True)
            names = DIR_NAMES.get(subdir)
            if names is None:
                names = E.names(E.name(subdir, lang="en"))
            root = E(
                "qet-directory",
                names,
            )
            with open(os.path.join(directory, "qet_directory"), "w") as fp:
                fp.write(etree.tostring(root, encoding="unicode", pretty_print=True))
        except OSError:
            pass
    return directory


def getter(c: Component) -> list[str]:
    """Returns a list of subdirs where the device should go into."""
    return c.get_fields("function poles connection spd_type curve Inc rc_type Idn In section more")


def main() -> None:
    QtGui.QGuiApplication()
    components: list[Component] = []

    for filename in tqdm(ARGS.input, unit="files", desc="Reading files      "):
        with open(filename, "r") as fp:
            for line in fp:
                line = line.strip()
                line = re.sub(r"\s+", line, " ")
                designation, description = line.split(maxsplit=1)
                if not description:
                    continue
                if not Component.accept(description):
                    continue
                try:
                    data: dict[str, Any] = {
                        "designation": designation,
                        "description": description,
                    }
                    description = description.upper()
                    claimed = False
                    if not claimed:
                        claimed = MCB.read(components, data, designation, description)
                    if not claimed:
                        claimed = RCCB.read(components, data, designation, description)
                    if not claimed:
                        claimed = AFDD.read(components, data, designation, description)
                    if not claimed:
                        claimed = RCBO.read(components, data, designation, description)
                    if not claimed:
                        claimed = AFDB.read(components, data, designation, description)
                    if not claimed:
                        claimed = Terminal.read(components, data, designation, description)
                    if not claimed:
                        claimed = SPD.read(components, data, designation, description)
                    if not claimed:
                        print(f"Nobody claimed {line}")
                except (KeyError, IndexError, ValueError) as exc:
                    print(f"Error {exc} in: {line}")

    build_dir = directory(os.path.expanduser("~/.qet/elements/"), ["siemens"])

    for path_components, device_group in itertools.groupby(
        tqdm(sorted(components, key=getter), unit="comps", desc="Building Components"), getter
    ):
        device_dir = directory(build_dir, path_components)
        for component in device_group:
            filename = f"{component.data["designation"]}.elmt"
            try:
                root = component.draw()
                with open(os.path.join(device_dir, filename), "w") as fp:
                    fp.write(
                        etree.tostring(root, encoding="unicode", pretty_print=True)
                    )
            except (KeyError, ZeroDivisionError):
                print(component.data)
                raise


def build_parser() -> argparse.ArgumentParser:
    """Build the commandline parser"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,  # don't wrap my description
        description=__doc__,
    )

    parser.add_argument("-v", "--verbose", action="store_true")

    parser.add_argument(
        "input",
        nargs="+",
        metavar="INPUT",
        help="The input files (downloaded with scrape.py)",
    )

    return parser


if __name__ == "__main__":

    parser = build_parser()
    parser.parse_args(namespace=ARGS)
    main()
