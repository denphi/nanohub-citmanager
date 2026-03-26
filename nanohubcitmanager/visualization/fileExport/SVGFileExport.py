from __future__ import annotations

import html
import math
from typing import Dict

from .GraphFileExport import GraphFileExport
from nanohubcitmanager.visualization.network.Vertex import Vertex
from nanohubcitmanager.visualization.network.Edge import Edge
from nanohubcitmanager.visualization.network.WeightedLabelledEdge import WeightedLabelledEdge


class SVGFileExport(GraphFileExport):
    def __init__(self, n):
        super().__init__(n)
        self.outW = 0.0
        self.outH = 0.0
        self.outD = 0.0
        self.oriW = float(self.network.canvasWidth)
        self.oriH = float(self.network.canvasHeight)
        self.oriD = float(math.hypot(self.oriW, self.oriH))
        self.graphW = 0.0
        self.graphH = 0.0
        self.graphD = 0.0
        self.titleHeight = 0.0
        self.legendHeight = 0.0
        self.bottomTextHeight = 0.0
        self.hMargin = 0.0

        self.legend: dict[str, tuple[int, int, int]] = {}
        self.sizeLegend: dict[float, str] = {}
        self.title = ""
        self.bottomText = ""
        self.backgroundColor = (255, 255, 255)

    def setBackgroundColor(self, bg: tuple[int, int, int]):
        self.backgroundColor = bg

    def toWebColor(self, c: tuple[int, int, int]) -> str:
        return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"

    def drawBackground(self) -> str:
        r, g, b = self.backgroundColor
        return (
            f'<rect x="0" y="0" width="{self.outW}" height="{self.outH}" '
            f'style="fill:rgb({r},{g},{b});stroke-width:0;"/>'
        )

    def drawNodes(self) -> str:
        nStr = ""
        for v in list(self.network.currentGraph.nodes()):
            wlv = self.network.getVerticesInfo().get(v)
            if wlv is None:
                continue
            fillOpacity = wlv.fillOpacity
            if fillOpacity < 0.01:
                continue
            strokeColor = self.toWebColor(wlv.borderColor)
            strokeOpacity = wlv.borderOpacity
            fillColor = self.toWebColor(wlv.fillColor)
            radius = wlv.radius * self.outD / self.oriD
            x = self.hMargin + wlv.coordinate.getX() / self.oriW * self.graphW
            y = self.titleHeight + wlv.coordinate.getY() / self.oriH * self.graphH
            node = (
                f'<circle stroke="{strokeColor}" stroke-opacity="{strokeOpacity}" '
                f'fill="{fillColor}" fill-opacity="{fillOpacity}" r="{radius}" '
                f'cy="{y}" cx="{x}" id="{v}"/>\n'
            )
            nStr += node
        return nStr

    def drawEdges(self) -> str:
        eStr = ""
        weightThreshold = 0
        for _, entry in self.network.getEdgesInfo().items():
            wle = entry
            fromV = wle.fromV
            toV = wle.toV
            opacity = wle.opacity
            if opacity < 0.01:
                continue

            if (
                fromV.extraProperties.get("NCN") is not None
                and int(fromV.extraProperties.get("NCN", 0)) > 0
                and int(toV.extraProperties.get("NCN", 0)) > 0
                and opacity > 0.9
            ):
                opacity = 0.2

            fromX = self.hMargin + fromV.coordinate.getX() * self.graphW / self.oriW
            fromY = self.titleHeight + fromV.coordinate.getY() * self.graphH / self.oriH
            toX = self.hMargin + toV.coordinate.getX() * self.graphW / self.oriW
            toY = self.titleHeight + toV.coordinate.getY() * self.graphH / self.oriH

            strokeColor = "#" + format((wle.color[0] << 16) + (wle.color[1] << 8) + wle.color[2], "x")
            weight = wle.weight
            lineWidth = int(wle.lineWidth * self.graphD / self.oriD) * (weight // 1)
            if weight <= weightThreshold:
                continue
            edge = (
                f'<line x1="{fromX}" y1="{fromY}" x2="{toX}" y2="{toY}" '
                f"style='stroke:{strokeColor};stroke-width:{lineWidth};stroke-opacity:{opacity}'/>\n"
            )
            eStr += edge
        return eStr

    def drawTitle(self) -> str:
        fontSize = self.outW / 30
        r, g, b = self.backgroundColor
        fontColor = (255 - r, 255 - g, 255 - b)
        text = (
            f'<text x="{self.outW/2}" y="{self.titleHeight/2}" font-size="{fontSize}" '
            f'fill="{self.toWebColor(fontColor)}" text-anchor="middle">{html.escape(self.title)}</text>'
        )
        return text

    def drawBottomText(self) -> str:
        fontSize = self.outW / 90
        r, g, b = self.backgroundColor
        fontColor = (255 - r, 255 - g, 255 - b)
        text = (
            f'<text x="{self.outW/2}" y="{self.outH - self.bottomTextHeight/2}" font-size="{fontSize}" '
            f'fill="{self.toWebColor(fontColor)}" text-anchor="middle">{html.escape(self.bottomText)}</text>'
        )
        return text

    def drawSizeLegend(self) -> str:
        text = ""
        num = len(self.sizeLegend)
        i = 0
        if num == 0:
            return text

        step = self.outW / num
        fontSize = self.outW / 90
        r, g, b = self.backgroundColor
        fontColor = (255 - r, 255 - g, 255 - b)

        text += (
            f'<text x="{self.legendHeight*1.5}" y="{self.outH - self.bottomTextHeight / 16 * 12}" '
            f'font-size="{fontSize}" fill="{self.toWebColor(fontColor)}" text-anchor="start">'
            'Three distinct dot sizes indicate the level of influence on h-index</text>'
        )

        for radius_key, txt in sorted(self.sizeLegend.items(), key=lambda kv: kv[0]):
            radius = radius_key * self.outD / self.oriD
            if radius < 2:
                continue
            text += (
                f'<circle cx="{i*step + self.legendHeight}" cy="{self.outH - self.bottomTextHeight / 2}" '
                f'r="{radius}" fill="{self.toWebColor(fontColor)}"/>'
            )
            text += (
                f'<text x="{i*step + self.legendHeight*1.5}" y="{self.outH - self.bottomTextHeight / 16 * 7}" '
                f'font-size="{fontSize}" fill="{self.toWebColor(fontColor)}" text-anchor="start">{html.escape(txt)}</text>'
            )
            i += 1
        return text

    def addTitle(self, t: str):
        self.title = t

    def addLegend(self, labelColor: dict[str, tuple[int, int, int]]):
        self.legend = dict(labelColor)

    def addSizeLegend(self, sl: list[str]):
        rValues: list[float] = []
        for v in list(self.network.currentGraph.nodes()):
            wlv = self.network.getVerticesInfo().get(v)
            if wlv is None:
                continue
            radius = float(wlv.radius)
            if radius not in rValues:
                rValues.append(radius)
        rValues.sort()

        i = 0
        for d in rValues:
            txt = sl[i] if i < len(sl) else ""
            self.sizeLegend[d] = txt
            i += 1

    def addBottomText(self, text: str):
        self.bottomText = text

    def drawLegend(self) -> str:
        text = ""
        num = len(self.legend)
        if num == 0:
            return text
        i = 0
        step = self.outW / num
        fontSize = self.outW / 60
        for label, c in sorted(self.legend.items(), key=lambda kv: kv[0]):
            text += (
                f'<circle cx="{i*step + self.legendHeight}" cy="{self.outH - self.legendHeight/2 - self.bottomTextHeight}" '
                f'r="{self.legendHeight/4}" fill="{self.toWebColor(c)}"/>\n'
            )
            text += (
                f'<text x="{i*step + self.legendHeight*1.5}" y="{self.outH - self.legendHeight/8*3 - self.bottomTextHeight}" '
                f'font-size="{fontSize}" fill="{self.toWebColor(c)}" text-anchor="start">{html.escape(label)}</text>\n'
            )
            i += 1
        return text

    def exportSVG(self, w: int, h: int) -> str:
        self.outW = float(w)
        self.outH = float(h)
        self.titleHeight = self.outH / 10
        self.legendHeight = self.outH / 20
        self.bottomTextHeight = self.outH / 20
        self.hMargin = self.outW / 50
        self.graphW = self.outW - self.hMargin * 2
        self.graphH = self.outH - self.titleHeight - self.legendHeight - self.bottomTextHeight
        self.outD = float(math.hypot(w, h))
        self.graphD = float(math.hypot(self.graphW, self.graphH))

        all_svg = (
            '<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
            '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" '
            '"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n'
            f'<svg height="{self.outH}px" width="{self.outW}px" version="1.1" '
            'xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">\n'
        )
        all_svg += self.drawBackground()
        all_svg += self.drawTitle()
        all_svg += self.drawEdges()
        all_svg += self.drawNodes()
        all_svg += self.drawLegend()
        all_svg += self.drawSizeLegend()
        all_svg += "</svg>"
        return all_svg

    def export(self, fileName: str, w: int, h: int):
        self.writeToFile(fileName, self.exportSVG(w, h), False)

    def sendSVGViaJSONRPC(self, fileName: str, w: int, h: int, url: str):
        self.export(fileName, w, h)

    def print(self, fileName: str, w: int, h: int, url: str):
        print(self.exportSVG(w, h))
