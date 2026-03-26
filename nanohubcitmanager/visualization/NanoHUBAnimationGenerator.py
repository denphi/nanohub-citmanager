from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import subprocess
import shutil

try:
    import cairosvg
except Exception:  # pragma: no cover
    cairosvg = None

try:  # Optional; used by the updated data-loading flow.
    from nanohubremote import Session
    from nanohubcitmanager import CitationManagerClient
except Exception:  # pragma: no cover
    Session = None
    CitationManagerClient = None

from nanohubcitmanager.visualization.network.NanoHUBCoDocumentNetwork import NanoHUBCoDocumentNetwork
from nanohubcitmanager.visualization.fileExport.SVGFileExport import SVGFileExport


class NanoHUBAnimationGenerator:
    def __init__(
        self,
        bgnYear: int,
        endYear: int,
        inputPrefix: str | None = None,
        outDir: str = ".",
        force_refresh: bool = False,
    ):
        self.outDir = Path(outDir)
        self.outDir.mkdir(parents=True, exist_ok=True)
        self._run_start = time.time()
        self._total_exports = 0
        self._export_index = 0
        self.createCoDocumentNetwork(
            bgnYear,
            endYear,
            inputPrefix=inputPrefix,
            force_refresh=force_refresh,
        )

    def _out(self, name: str) -> str:
        return str(self.outDir / name)

    def _log(self, message: str):
        elapsed = time.time() - self._run_start
        print(f"[{elapsed:7.1f}s] {message}", flush=True)

    def _export_svg(
        self,
        fe: SVGFileExport,
        file_name: str,
        outputW: int,
        outputH: int,
        saveSVGURL: str,
    ):
        self._export_index += 1
        self._log(f"Export {self._export_index}/{self._total_exports}: {file_name}")
        fe.sendSVGViaJSONRPC(self._out(file_name), outputW, outputH, saveSVGURL)

    def createCoDocumentNetwork(
        self,
        bgnYear: int,
        endYear: int,
        inputPrefix: str | None = None,
        force_refresh: bool = False,
    ):
        self._log(f"Starting generation for {bgnYear}-{endYear} (output: {self.outDir})")

        if inputPrefix:
            self._log(f"Loading network from prefix: {inputPrefix}")
            network = NanoHUBCoDocumentNetwork(inFile=inputPrefix)
        else:
            token = os.environ.get("NANOHUB_TOKEN")
            hub_url = os.environ.get("NANOHUB_URL", "https://nanohub.org")
            if not token:
                raise RuntimeError("NANOHUB_TOKEN is not set (checked environment and .env in current directory)")
            if Session is None or CitationManagerClient is None:
                raise RuntimeError(
                    "nanohubremote/nanohubcitmanager are required for DocumentNetwork loading"
                )

            self._log(f"Connecting to nanoHUB at {hub_url}")
            session = Session(
                {"grant_type": "personal_token", "token": token},
                url=hub_url,
                timeout=120,
            )
            client = CitationManagerClient(session)

            network = NanoHUBCoDocumentNetwork(
                client=client,
                begin_year=bgnYear,
                end_year=endYear,
                cache_dir=str(self.outDir),
                force_refresh=force_refresh,
            )
            self._log("Fetching document data")
            network.fetch()
            self._log("Processing document data into graph")
            network.process()

        self._log(
            "Graph ready: "
            f"{network.getGraph().number_of_nodes()} nodes, "
            f"{network.getGraph().number_of_edges()} edges"
        )
        self._log("Computing component sizes and network IDs")
        network.computeNodeComponentSize()
        network.computeNetworkID()

        hIndex = int(network.getExtraInfo().get("h-index", 0))
        citRange = [hIndex - 10, hIndex]
        self._log(f"Computed h-index={hIndex}, citation range={citRange}")

        network.sizeByRange("citations", citRange)
        self._log("Running Dust&Magnet layout")
        network.setLayout("Dust&Magnet", 5, use_kamada_kawai=True)
        self._log("Layout finished")

        by = bgnYear
        year_count = (endYear - by + 1)
        self._total_exports = year_count * 24
        self._log(f"Preparing to export {self._total_exports} SVG files across {year_count} year windows")
        for ey in range(by, endYear + 1):
            self._log(f"Year window {by}-{ey}: applying filters and exporting")
            outFile = f"{by}-{ey}"
            network.edgeOpacityBy(1)
            network.setInYearRange(by, ey)
            network.sizeByRange("citations", citRange)
            self.outputGraph(network, outFile, by, ey, "-var-size")
            network.sizeByBinary("citations", 9999999)
            self.outputGraph(network, outFile, by, ey, "-same-size")
        self._log("SVG generation complete")

    def outputGraph(self, network: NanoHUBCoDocumentNetwork, outFile: str, by: int, ey: int, label: str):
        outputW = 2000
        outputH = 1725
        sizeLegend = [
            "Papers with relatively low secondary citations",
            "Papers with potential to influence h-index",
            "Papers affecting the h-index",
        ]
        saveSVGURL = "http://nanohub.org/nanoHUBCitations/com_citmanager/NCN_report/saveSVG.php"

        network.colorNodeBy("NCN-affiliated", 0)
        network.opacityBy("inYearRange", 0, 1, True)
        feNCN = SVGFileExport(network)
        feNCN.setBackgroundColor((0, 0, 0))
        feNCN.addTitle(f"NCN vs. Non-NCN ({by}-{ey})")
        legend = {
            "NCN-affiliated documents": network.getColor(1),
            "Non-NCN-affiliated documents": network.getColor(0),
        }
        feNCN.addLegend(legend)
        if label == "-var-size":
            feNCN.addSizeLegend(sizeLegend)
        self._export_svg(feNCN, f"{outFile}-NCN-black{label}.svg", outputW, outputH, saveSVGURL)

        network.colorNodeBy("NCN-affiliated", 16)
        feNCN.setBackgroundColor((255, 255, 255))
        legend = {
            "NCN-affiliated documents": network.getColor(17),
            "Non-NCN-affiliated documents": network.getColor(16),
        }
        feNCN.addLegend(legend)
        if label == "-var-size":
            feNCN.addSizeLegend(sizeLegend)
        self._export_svg(feNCN, f"{outFile}-NCN-white{label}.svg", outputW, outputH, saveSVGURL)
        network.edgeOpacityBy(0)

        network.colorNodeBy("NCN-affiliated", 0)
        feNCN.setBackgroundColor((0, 0, 0))
        legend = {
            "NCN-affiliated documents": network.getColor(1),
            "Non-NCN-affiliated documents": network.getColor(0),
        }
        feNCN.addLegend(legend)
        if label == "-var-size":
            feNCN.addSizeLegend(sizeLegend)
        self._export_svg(feNCN, f"{outFile}-NCN-NodeOnly-black{label}.svg", outputW, outputH, saveSVGURL)

        network.colorNodeBy("NCN-affiliated", 16)
        feNCN.setBackgroundColor((255, 255, 255))
        legend = {
            "NCN-affiliated documents": network.getColor(17),
            "Non-NCN-affiliated documents": network.getColor(16),
        }
        feNCN.addLegend(legend)
        if label == "-var-size":
            feNCN.addSizeLegend(sizeLegend)
        self._export_svg(feNCN, f"{outFile}-NCN-NodeOnly-white{label}.svg", outputW, outputH, saveSVGURL)
        network.opacityBy("inYearRange", 0, 1, True)

        network.colorNodeBy("refTypeCode", 2)
        feRT = SVGFileExport(network)
        feRT.setBackgroundColor((0, 0, 0))
        feRT.addTitle(f"Reference Types ({by}-{ey})")
        legend = {
            "Research and Education": network.getColor(2),
            "Cyberinfrastructure": network.getColor(3),
            "Education": network.getColor(4),
            "Research": network.getColor(5),
        }
        if label == "-var-size":
            feRT.addSizeLegend(sizeLegend)
        feRT.addLegend(legend)
        self._export_svg(feRT, f"{outFile}-RefTypeCode-black{label}.svg", outputW, outputH, saveSVGURL)

        network.colorNodeBy("refTypeCode", 18)
        feRT.setBackgroundColor((255, 255, 255))
        legend = {
            "Research and Education": network.getColor(18),
            "Cyberinfrastructure": network.getColor(19),
            "Education": network.getColor(20),
            "Research": network.getColor(21),
        }
        feRT.addLegend(legend)
        if label == "-var-size":
            feRT.addSizeLegend(sizeLegend)
        self._export_svg(feRT, f"{outFile}-RefTypeCode-white{label}.svg", outputW, outputH, saveSVGURL)

        network.colorNodeBy("same", 15)
        feSame = SVGFileExport(network)
        feSame.setBackgroundColor((0, 0, 0))
        if label == "-var-size":
            feSame.addSizeLegend(sizeLegend)
        feSame.addTitle(f"nanoHUB Citation Network ({by}-{ey})")
        self._export_svg(feSame, f"{outFile}-White-black{label}.svg", outputW, outputH, saveSVGURL)

        network.colorNodeBy("same", 6)
        feSame.setBackgroundColor((255, 255, 255))
        if label == "-var-size":
            feSame.addSizeLegend(sizeLegend)
        self._export_svg(feSame, f"{outFile}-Black-white{label}.svg", outputW, outputH, saveSVGURL)

        network.colorNodeBy("toolCode", 10)
        feTC = SVGFileExport(network)
        feTC.setBackgroundColor((0, 0, 0))
        feTC.addTitle(f"Tool Usage ({by}-{ey})")
        legend = {
            "Workspace": network.getColor(10),
            "Schred": network.getColor(11),
            "nanoMOS": network.getColor(12),
            "FETToy": network.getColor(13),
            "Others": network.getColor(15),
        }
        feTC.addLegend(legend)
        if label == "-var-size":
            feTC.addSizeLegend(sizeLegend)
        self._export_svg(feTC, f"{outFile}-Tool-black{label}.svg", outputW, outputH, saveSVGURL)

        network.colorNodeBy("toolCode", 26)
        feTC.setBackgroundColor((255, 255, 255))
        legend = {
            "Workspace": network.getColor(26),
            "Schred": network.getColor(27),
            "nanoMOS": network.getColor(28),
            "FETToy": network.getColor(29),
            "Others": network.getColor(31),
        }
        feTC.addLegend(legend)
        if label == "-var-size":
            feTC.addSizeLegend(sizeLegend)
        self._export_svg(feTC, f"{outFile}-Tool-white{label}.svg", outputW, outputH, saveSVGURL)

        network.colorNodeBy("expListDataCode", 7)
        network.opacityBy("refTypeAllRes", 0, 1, False)
        feED = SVGFileExport(network)
        feED.setBackgroundColor((0, 0, 0))
        legend = {
            "Exp. data": network.getColor(8),
            "Experimentalist and Exp. data": network.getColor(9),
            "Non-Experimental": network.getColor(7),
        }
        feED.addLegend(legend)
        if label == "-var-size":
            feED.addSizeLegend(sizeLegend)
        feED.addTitle(f"Experimentalist and Experimental Data ({by}-{ey})")
        self._export_svg(feED, f"{outFile}-ExpData-black{label}.svg", outputW, outputH, saveSVGURL)

        network.colorNodeBy("expListDataCode", 23)
        feED.setBackgroundColor((255, 255, 255))
        legend = {
            "Exp. data": network.getColor(24),
            "Experimentalist and Exp. data": network.getColor(25),
            "Non-Experimental": network.getColor(23),
        }
        feED.addLegend(legend)
        if label == "-var-size":
            feED.addSizeLegend(sizeLegend)
        self._export_svg(feED, f"{outFile}-ExpData-white{label}.svg", outputW, outputH, saveSVGURL)


def rasterize_svg(svg_path: Path):
    png_path = svg_path.with_suffix(".png")
    if cairosvg is not None:
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))
    elif shutil.which("sips"):
        subprocess.run(
            ["sips", "-s", "format", "png", str(svg_path), "--out", str(png_path)],
            check=True, capture_output=True,
        )
    else:
        raise RuntimeError("No SVG rasterizer found: install cairosvg or run on macOS (sips)")


def main():
    # Load .env from the current working directory if present.
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    current_year = time.localtime().tm_year

    parser = argparse.ArgumentParser()
    parser.add_argument("begin_year", type=int, nargs="?", default=2000)
    parser.add_argument("end_year", type=int, nargs="?", default=current_year)
    parser.add_argument("--input-file", type=str, default=None, help="Path to a documents_*.json cache file")
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--no-rasterize", action="store_true")
    args = parser.parse_args()

    NanoHUBAnimationGenerator(
        args.begin_year,
        args.end_year,
        inputPrefix=args.input_file,
        outDir=args.out_dir,
        force_refresh=args.force_refresh,
    )

    if not args.no_rasterize:
        if cairosvg is None and not shutil.which("sips"):
            raise RuntimeError("--rasterize requires cairosvg (pip install cairosvg) or macOS sips")
        out_dir = Path(args.out_dir)
        svgs = sorted(out_dir.glob("*.svg"))
        total = len(svgs)
        print(f"[   0.0s] Rasterizing {total} SVG files", flush=True)
        start = time.time()
        for i, svg in enumerate(svgs, start=1):
            if i == 1 or i % 10 == 0 or i == total:
                elapsed = time.time() - start
                print(f"[{elapsed:7.1f}s] Rasterize {i}/{total}: {svg.name}", flush=True)
            rasterize_svg(svg)


if __name__ == "__main__":
    main()
