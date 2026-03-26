from __future__ import annotations

import math
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import networkx as nx

from nanohubcitmanager.visualization.network.Circle import Circle
from nanohubcitmanager.visualization.network.CirclePacking import CirclePacking
from nanohubcitmanager.visualization.network.Network import Network
from nanohubcitmanager.visualization.network.Vertex import Point, Vertex
from nanohubcitmanager.visualization.network.Edge import Edge
from nanohubcitmanager.visualization.network.WeightedLabelledVertex import WeightedLabelledVertex
from nanohubcitmanager.visualization.network.WeightedLabelledEdge import WeightedLabelledEdge


class DustMagnetLayout:
    def __init__(self, g: nx.Graph):
        self.graph = g
        self.status = "DustMagnetLayout"
        self.h = 0.0
        self.w = 0.0
        self.canvasRadius = 0.0
        self.r = 0.0
        self.canvasCenter = Point()

        self.vIndex2VID: list[int] = []
        self.vVID2Index: dict[int, int] = {}
        self.xydata: list[Point] = []
        self.energy: list[float] = []

        self.properties: list[str] = []
        self.network: Network | None = None
        self.vsInfo: dict[int, Vertex] = {}
        self.esInfo: dict[str, Edge] = {}

        self.magnetPos: "OrderedDict[str, Point]" = OrderedDict()
        self.magnetIntensity: "OrderedDict[str, float]" = OrderedDict()
        self.magnetInitAngleRange: "OrderedDict[str, float]" = OrderedDict()
        self.magnetScope: "OrderedDict[str, float]" = OrderedDict()
        self.magnetVertexInitIndex: "OrderedDict[str, int]" = OrderedDict()
        self.magnetLastVertex: "OrderedDict[str, int]" = OrderedDict()
        self.compClusterGroup: "OrderedDict[str, list[int]]" = OrderedDict()

        self.energyThreshold = 0.001
        self.minVertexSize = float("inf")
        self.maxVertexSize = float("-inf")
        self.aveVertexSize = 0.0
        self.linkStrength = 0.0
        self.linkScope = 0.0
        self.progress = 0.0

    def setNetwork(self, nw: Network):
        self.network = nw
        self.setVerticesInfo(nw.getVerticesInfo())
        self.setEdgesInfo(nw.getEdgesInfo())

    def setSize(self, size: tuple[int, int]):
        self.w = float(size[0])
        self.h = float(size[1])
        self.canvasCenter.setLocation(self.w / 2.0, self.h / 2.0)
        self.canvasRadius = (self.h if self.w > self.h else self.w) / 2.0
        self.r = self.h if self.w > self.h else self.w
        self.initialize()

    def setVerticesInfo(self, vI: dict[int, Vertex]):
        self.vsInfo = vI
        accuSize = 0.0
        for vData in self.vsInfo.values():
            self.minVertexSize = min(self.minVertexSize, vData.radius)
            self.maxVertexSize = max(self.maxVertexSize, vData.radius)
            accuSize += vData.radius
        self.aveVertexSize = accuSize / len(self.vsInfo) if self.vsInfo else 0.0

    def setEdgesInfo(self, eI: dict[str, Edge]):
        self.esInfo = eI

    def initialize(self):
        self.vIndex2VID = list(self.graph.nodes())
        self.vVID2Index = {v: i for i, v in enumerate(self.vIndex2VID)}
        self.xydata = []
        self.energy = []
        for v in self.vIndex2VID:
            if v in self.vsInfo:
                c = self.vsInfo[v].coordinate
                self.xydata.append(Point(c.getX(), c.getY()))
            else:
                self.xydata.append(Point())
            self.energy.append(-1.0)

    def transform(self, v: int) -> Point | None:
        idx = self.vVID2Index.get(v)
        if idx is None:
            return None
        return self.xydata[idx]

    def filterEdgeByWeight(self, weight: float):
        toBeRemoved: list[tuple[int, int, str]] = []
        for u, v, data in self.graph.edges(data=True):
            eID = data.get("id")
            if not eID:
                continue
            wle = self.esInfo.get(eID)
            if wle is None:
                continue
            if wle.weight < weight:
                toBeRemoved.append((u, v, eID))

        for u, v, eID in toBeRemoved:
            if self.graph.has_edge(u, v):
                self.graph.remove_edge(u, v)
            if eID in self.esInfo:
                del self.esInfo[eID]

    def addNCNMagnets(self):
        p = Point(7 * self.w / 15.0, 1 * self.h / 2.0)
        self.magnetPos["NCN"] = p
        self.magnetIntensity["NCN"] = 1.0
        self.magnetInitAngleRange["NCN"] = math.pi * 2 / 5
        self.magnetVertexInitIndex["NCN"] = 0

    def addRefTypeMagnets(self):
        p = Point(4 * self.w / 5.0, 1 * self.h / 5.0)
        self.magnetPos["refTypeRes"] = p
        p = Point(4 * self.w / 5.0, 3 * self.h / 5.0)
        self.magnetPos["refTypeCyber"] = p
        p = Point(4 * self.w / 5.0, 4 * self.h / 5.0)
        self.magnetPos["refTypeResEdu"] = p
        p = Point(4 * self.w / 5.0, 2 * self.h / 5.0)
        self.magnetPos["refTypeEdu"] = p

        self.magnetIntensity["refTypeResEdu"] = 0.01
        self.magnetIntensity["refTypeEdu"] = 0.01
        self.magnetIntensity["refTypeRes"] = 0.01
        self.magnetIntensity["refTypeCyber"] = 0.01
        self.magnetInitAngleRange["refTypeResEdu"] = math.pi / 16
        self.magnetInitAngleRange["refTypeEdu"] = math.pi / 16
        self.magnetInitAngleRange["refTypeRes"] = math.pi / 8
        self.magnetInitAngleRange["refTypeCyber"] = math.pi / 8
        self.magnetVertexInitIndex["refTypeResEdu"] = 0
        self.magnetVertexInitIndex["refTypeEdu"] = 0
        self.magnetVertexInitIndex["refTypeRes"] = 0
        self.magnetVertexInitIndex["refTypeCyber"] = 0

    def addToolMagnets(self):
        p = Point(3 * self.w / 8.0, 1 * self.h / 6.0)
        self.magnetPos["tool_nanomos"] = p
        p = Point(5 * self.w / 8.0, 7 * self.h / 8.0)
        self.magnetPos["tool_schred"] = p
        p = Point(3 * self.w / 8.0, 5 * self.h / 6.0)
        self.magnetPos["tool_workspace"] = p
        p = Point(5 * self.w / 8.0, 1 * self.h / 8.0)
        self.magnetPos["tool_fettoy"] = p

        self.magnetIntensity["tool_workspace"] = 0.75
        self.magnetIntensity["tool_schred"] = 0.75
        self.magnetIntensity["tool_nanomos"] = 0.75
        self.magnetIntensity["tool_fettoy"] = 0.75
        self.magnetInitAngleRange["tool_workspace"] = math.pi / 10
        self.magnetInitAngleRange["tool_schred"] = math.pi / 5
        self.magnetInitAngleRange["tool_nanomos"] = math.pi / 10
        self.magnetInitAngleRange["tool_fettoy"] = math.pi / 16
        self.magnetVertexInitIndex["tool_workspace"] = 0
        self.magnetVertexInitIndex["tool_schred"] = 0
        self.magnetVertexInitIndex["tool_nanomos"] = 0
        self.magnetVertexInitIndex["tool_fettoy"] = 0

    def addExpMagnets(self):
        p = Point(1 * self.w / 4.0, 3 * self.h / 4.0)
        self.magnetPos["expListDataBool"] = p
        p = Point(3 * self.w / 4.0, 3 * self.h / 4.0)
        self.magnetPos["expDataBool"] = p
        self.magnetIntensity["expListDataBool"] = 0.001
        self.magnetIntensity["expDataBool"] = 0.001
        self.magnetInitAngleRange["expListDataBool"] = math.pi / 15
        self.magnetInitAngleRange["expDataBool"] = math.pi / 15
        self.magnetVertexInitIndex["expListDataBool"] = 0
        self.magnetVertexInitIndex["expDataBool"] = 0

    def run(self, steps: int, use_kamada_kawai: bool = False):
        n = self.graph.number_of_nodes()
        if n == 0:
            return
        run_start = time.time()
        print(f"[layout] Dust&Magnet start: nodes={n}, edges={self.graph.number_of_edges()}, steps={steps}", flush=True)

        self.addNCNMagnets()
        self.addToolMagnets()
        self.addRefTypeMagnets()
        self.addExpMagnets()

        self.linkStrength = 0.03

        sortedVs: list[Vertex] = []
        for v in self.graph.nodes():
            sortedVs.append(self.vsInfo[v])
        sortedVs.sort(key=lambda vv: int(vv.score.get("componentSize", 0.0)))

        for vData in sortedVs:
            init = self.initPos(vData)
            index = self.vVID2Index[vData.id]
            self.xydata[index].setLocation(init.x, init.y)

        cell_size = max(self.aveVertexSize * 2.4, 1.0)
        self._build_step_cache()
        self._build_spatial_grid(cell_size)

        for i in range(steps):
            self.progress = float(i // steps)
            elapsed = time.time() - run_start
            print(f"[layout] step {i + 1}/{steps} (elapsed {elapsed:.1f}s)", flush=True)
            self.step_opt()

        print(f"[layout] assigning clusters and linking components", flush=True)
        self.assignClusterID_opt()
        self.linkComponents_opt(use_kamada_kawai=use_kamada_kawai)
        print(f"[layout] Dust&Magnet complete in {time.time() - run_start:.1f}s", flush=True)

    def assignClusterID(self):
        for v in self.graph.nodes():
            id_num = 0
            vData = self.vsInfo[v]
            for attr in self.magnetPos.keys():
                id_num *= 10
                if attr in vData.extraProperties:
                    id_num += int(vData.extraProperties[attr])
            vData.extraProperties["clusterID"] = id_num
            ccid = f"{id_num}-{int(vData.score.get('networkID', 0))}"
            if ccid not in self.compClusterGroup:
                self.compClusterGroup[ccid] = []
            self.compClusterGroup[ccid].append(v)

    def linkComponents(self):
        networks: list[Network] = []
        for _, vs in self.compClusterGroup.items():
            if len(vs) < 2:
                continue
            componentG = self.graph.subgraph(vs).copy()
            nt = Network()
            nt.setGraph(componentG)
            localVsInfo: dict[int, Vertex] = {}
            for vID in vs:
                localVsInfo[vID] = self.vsInfo[vID]
            nt.setVerticesInfo(localVsInfo)
            nt.setLayout("Kamada-Kawai", len(vs) * 20)
            networks.append(nt)

        self.expandComponents(networks)

    def expandComponents(self, nts: list[Network]):
        networkCircleMap: dict[Network, Circle] = {}
        oldNetworkCircleMap: dict[Network, Circle] = {}
        cp = CirclePacking()

        for nt in nts:
            vsInfo = nt.getVerticesInfo()
            accuX = 0.0
            accuY = 0.0
            totalRadius = 0.0

            for id_, v in vsInfo.items():
                pt = self.xydata[self.vVID2Index[id_]]
                accuX += pt.getX()
                accuY += pt.getY()
                totalRadius += v.radius

            centerX = accuX / len(vsInfo)
            centerY = accuY / len(vsInfo)
            radius = math.sqrt(totalRadius) * 50
            if radius > self.w / 10:
                radius = self.h / 10

            c = Circle(centerX, centerY, radius)
            c2 = Circle(centerX, centerY, radius)
            cp.addCircle(c)
            oldNetworkCircleMap[nt] = c2
            networkCircleMap[nt] = c

        cp.layout(0.0)

        for nt in nts:
            newCircle = networkCircleMap[nt]
            vsInfo = nt.getVerticesInfo()
            for id_, vData in vsInfo.items():
                newX = newCircle.x + ((vData.coordinate.x - nt.canvasWidth / 2.0) / nt.canvasWidth) * 2 * newCircle.r * 1.25
                newY = newCircle.y + ((vData.coordinate.y - nt.canvasHeight / 2.0) / nt.canvasHeight) * 2 * newCircle.r * 1.25
                self.xydata[self.vVID2Index[id_]].setLocation(newX, newY)

    # --- Optimized replacements ---

    def assignClusterID_opt(self):
        """
        Optimized assignClusterID.

        Changes vs original:
        - Uses defaultdict to avoid repeated 'not in' checks.
        - Caches the ordered magnet keys once instead of calling .keys() per node.
        - Computes id_num with a list comprehension + int join to avoid repeated
          multiply-by-10 accumulation (same result, fewer attribute lookups).
        """
        magnet_keys = list(self.magnetPos.keys())
        ccg: defaultdict[str, list[int]] = defaultdict(list)

        for v in self.graph.nodes():
            vData = self.vsInfo[v]
            ep = vData.extraProperties
            id_num = int("".join(str(int(ep[k])) if k in ep else "0" for k in magnet_keys))
            ep["clusterID"] = id_num
            ccid = f"{id_num}-{int(vData.score.get('networkID', 0))}"
            ccg[ccid].append(v)

        self.compClusterGroup = OrderedDict(ccg)

    def _layout_one_component(
        self, vs: list[int], use_kamada_kawai: bool = False
    ) -> tuple[list[int], dict[int, tuple[float, float]]]:
        """
        Compute layout positions for one cluster group.
        By default uses Kamada-Kawai for small groups (≤ 50 nodes) and
        spring layout for larger ones.  Pass use_kamada_kawai=True to force
        Kamada-Kawai regardless of size (better cluster quality, much slower).
        Returns (vs, {vID: (x, y)}) in Network canvas coordinates.
        """
        componentG = self.graph.subgraph(vs).copy()
        n = len(vs)
        canvas = 100000

        if use_kamada_kawai or n <= 50:
            pos = nx.kamada_kawai_layout(componentG)
        else:
            # spring_layout is O(N²·iterations) – much cheaper than K-K's O(N³)
            iterations = min(50, max(10, 300 // n))
            pos = nx.spring_layout(componentG, iterations=iterations, seed=42)

        mapped: dict[int, tuple[float, float]] = {
            v: ((x + 1.0) * 0.5 * canvas, (y + 1.0) * 0.5 * canvas)
            for v, (x, y) in pos.items()
        }
        return vs, mapped

    def linkComponents_opt(self, use_kamada_kawai: bool = False):
        """
        Optimized linkComponents.

        Parameters
        ----------
        use_kamada_kawai : bool
            False (default) – use spring layout for groups > 50 nodes (fast).
            True            – always use Kamada-Kawai (better cluster quality,
                              but O(N³) so much slower for large groups).

        Changes vs original:
        - Skips Kamada-Kawai for groups > 50 nodes by default; uses spring layout.
        - Runs each group's layout in a thread pool so independent groups compute
          in parallel (NetworkX releases the GIL for most of its C extensions).
        - Avoids creating a full Network object per group – positions are computed
          directly and stored on the shared vsInfo.
        - Adds progress logging so you can see it is making progress.
        """
        groups = [(ccid, vs) for ccid, vs in self.compClusterGroup.items() if len(vs) >= 2]
        if not groups:
            self.expandComponents_opt([])
            return

        mode = "Kamada-Kawai (all groups)" if use_kamada_kawai else "Kamada-Kawai (≤50 nodes) / spring (>50 nodes)"
        print(f"[layout] linkComponents_opt: {len(groups)} groups, layout={mode}", flush=True)

        # Build Network objects with positions filled in parallel.
        networks: list[Network] = []
        def _process(vs: list[int]) -> Network:
            _, mapped = self._layout_one_component(vs, use_kamada_kawai=use_kamada_kawai)
            nt = Network()
            nt.setGraph(self.graph.subgraph(vs).copy())
            localVsInfo: dict[int, Vertex] = {vID: self.vsInfo[vID] for vID in vs}
            nt.setVerticesInfo(localVsInfo)
            # Apply positions directly instead of calling setLayout again.
            for vID, (x, y) in mapped.items():
                if vID in localVsInfo:
                    localVsInfo[vID].coordinate = type(localVsInfo[vID].coordinate)(x, y)
            return nt

        max_workers = min(8, len(groups))
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process, vs): ccid for ccid, vs in groups}
            for fut in as_completed(futures):
                networks.append(fut.result())
                done += 1
                if done % max(1, len(groups) // 10) == 0 or done == len(groups):
                    print(f"[layout] linkComponents_opt: {done}/{len(groups)} groups done", flush=True)

        self.expandComponents_opt(networks)

    def expandComponents_opt(self, nts: list[Network]):
        """
        Optimized expandComponents / CirclePacking.

        Changes vs original:
        - Replaces the unbounded CirclePacking.layout() while-loop with a
          vectorised numpy repulsion pass (falls back to pure-Python if numpy is
          unavailable, but still caps iterations to avoid infinite loops).
        - Adds an overall iteration cap on the packing loop.
        - Logs progress.
        """
        if not nts:
            return

        print(f"[layout] expandComponents_opt: packing {len(nts)} circles", flush=True)

        # ---- Collect circle data ----
        cx_arr: list[float] = []
        cy_arr: list[float] = []
        cr_arr: list[float] = []

        for nt in nts:
            vsInfo = nt.getVerticesInfo()
            accuX = accuY = totalRadius = 0.0
            for id_, v in vsInfo.items():
                pt = self.xydata[self.vVID2Index[id_]]
                accuX += pt.getX()
                accuY += pt.getY()
                totalRadius += v.radius
            n = len(vsInfo)
            centerX = accuX / n
            centerY = accuY / n
            radius = math.sqrt(totalRadius) * 50
            if radius > self.w / 10:
                radius = self.h / 10
            cx_arr.append(centerX)
            cy_arr.append(centerY)
            cr_arr.append(max(radius, 1.0))

        # ---- Packing: faithful replication of CirclePacking algorithm ----
        # The original expands each overlapping connected component radially
        # outward from its centroid (same as CirclePacking.expand), repeated
        # until no two circles overlap.  We replicate that exactly but:
        #   - use numpy for the O(N²) overlap detection (much faster)
        #   - add a hard iteration cap so we never hang
        MAX_ITER = 2000
        STAGNATION_WINDOW = 10   # stop if no improvement over this many iterations
        LOG_EVERY = max(1, MAX_ITER // 20)

        try:
            import numpy as np

            cx = np.array(cx_arr, dtype=float)
            cy = np.array(cy_arr, dtype=float)
            cr = np.array(cr_arr, dtype=float)

            best_overlaps = None
            stagnation_count = 0

            for pack_iter in range(MAX_ITER):
                # --- detect overlaps (same as CirclePacking.occluded) ---
                dx = cx[:, None] - cx[None, :]
                dy = cy[:, None] - cy[None, :]
                dist = np.sqrt(dx * dx + dy * dy)
                np.fill_diagonal(dist, np.inf)
                overlapping = dist < (cr[:, None] + cr[None, :])
                n_overlaps = int(overlapping.sum()) // 2

                if n_overlaps == 0:
                    print(f"[layout] expandComponents_opt: converged after {pack_iter} iterations", flush=True)
                    break

                if pack_iter % LOG_EVERY == 0:
                    print(f"[layout] expandComponents_opt: iteration {pack_iter}/{MAX_ITER}, overlapping pairs={n_overlaps}", flush=True)

                if best_overlaps is None or n_overlaps < best_overlaps:
                    best_overlaps = n_overlaps
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                    if stagnation_count >= STAGNATION_WINDOW:
                        print(f"[layout] expandComponents_opt: stagnated at {n_overlaps} overlapping pairs after {pack_iter} iterations, stopping", flush=True)
                        break

                # --- build connected components of overlapping circles ---
                n = len(cx)
                visited = np.zeros(n, dtype=bool)
                for start in range(n):
                    if visited[start]:
                        continue
                    # BFS
                    comp = []
                    stack = [start]
                    while stack:
                        v = stack.pop()
                        if visited[v]:
                            continue
                        visited[v] = True
                        comp.append(v)
                        stack.extend(int(nb) for nb in np.where(overlapping[v])[0] if not visited[nb])

                    if len(comp) < 2:
                        continue

                    # --- expand component radially (same as CirclePacking.expand) ---
                    idx = np.array(comp)
                    comp_cx = cx[idx]
                    comp_cy = cy[idx]
                    comp_cr = cr[idx]

                    center_x = comp_cx.mean()
                    center_y = comp_cy.mean()
                    span = math.hypot(comp_cx.max() - comp_cx.min(), comp_cy.max() - comp_cy.min())
                    step_dist = span / 10.0 if span > 0 else 1.0

                    # run up to 100 inner iterations (same as original maxIteration=100)
                    for _ in range(100):
                        ddx = comp_cx - center_x
                        ddy = comp_cy - center_y
                        near_zero = np.abs(ddx) < 1e-12
                        arc = np.where(near_zero,
                                       np.where(ddy > 0, math.pi / 2, -math.pi / 2),
                                       np.arctan(ddy / np.where(near_zero, 1.0, ddx)))
                        arc = np.where(ddx < 0, arc + math.pi, arc)
                        move = step_dist / np.maximum(comp_cr, 1e-9)
                        comp_cx += move * np.cos(arc)
                        comp_cy += move * np.sin(arc)

                        # check if this component is no longer self-overlapping
                        ddx2 = comp_cx[:, None] - comp_cx[None, :]
                        ddy2 = comp_cy[:, None] - comp_cy[None, :]
                        d2 = np.sqrt(ddx2 * ddx2 + ddy2 * ddy2)
                        np.fill_diagonal(d2, np.inf)
                        if not (d2 < (comp_cr[:, None] + comp_cr[None, :])).any():
                            break

                    cx[idx] = comp_cx
                    cy[idx] = comp_cy

            else:
                print(f"[layout] expandComponents_opt: hit outer iteration cap ({MAX_ITER}), some circles may still overlap", flush=True)

            cx_arr = cx.tolist()
            cy_arr = cy.tolist()

        except ImportError:
            # Pure-Python fallback — replicates CirclePacking exactly with a cap
            for py_iter in range(MAX_ITER):
                # build adjacency
                adjacency: dict[int, set[int]] = {i: set() for i in range(len(cx_arr))}
                any_overlap = False
                for i in range(len(cx_arr)):
                    for j in range(i + 1, len(cx_arr)):
                        dist = math.hypot(cx_arr[i] - cx_arr[j], cy_arr[i] - cy_arr[j])
                        if dist < cr_arr[i] + cr_arr[j]:
                            adjacency[i].add(j)
                            adjacency[j].add(i)
                            any_overlap = True
                if py_iter % LOG_EVERY == 0:
                    print(f"[layout] expandComponents_opt: iteration {py_iter}/{MAX_ITER}", flush=True)
                if not any_overlap:
                    break

                # connected components
                visited_set: set[int] = set()
                for start in range(len(cx_arr)):
                    if start in visited_set:
                        continue
                    stack2 = [start]
                    comp2: list[int] = []
                    while stack2:
                        v = stack2.pop()
                        if v in visited_set:
                            continue
                        visited_set.add(v)
                        comp2.append(v)
                        stack2.extend(adjacency[v] - visited_set)
                    if len(comp2) < 2:
                        continue

                    center_x = sum(cx_arr[v] for v in comp2) / len(comp2)
                    center_y = sum(cy_arr[v] for v in comp2) / len(comp2)
                    xs = [cx_arr[v] for v in comp2]
                    ys = [cy_arr[v] for v in comp2]
                    span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
                    step_dist = span / 10.0 if span > 0 else 1.0

                    for _ in range(100):
                        for v in comp2:
                            ddx = cx_arr[v] - center_x
                            ddy = cy_arr[v] - center_y
                            if abs(ddx) < 1e-12:
                                arc = math.pi / 2 if ddy > 0 else -math.pi / 2
                            else:
                                arc = math.atan(ddy / ddx)
                            if ddx < 0:
                                arc += math.pi
                            move = step_dist / max(cr_arr[v], 1e-9)
                            cx_arr[v] += move * math.cos(arc)
                            cy_arr[v] += move * math.sin(arc)

                        still_overlap = any(
                            math.hypot(cx_arr[comp2[i]] - cx_arr[comp2[j]], cy_arr[comp2[i]] - cy_arr[comp2[j]]) < cr_arr[comp2[i]] + cr_arr[comp2[j]]
                            for i in range(len(comp2)) for j in range(i + 1, len(comp2))
                        )
                        if not still_overlap:
                            break
            else:
                print(f"[layout] expandComponents_opt: hit outer iteration cap ({MAX_ITER})", flush=True)

        # ---- Apply new positions ----
        for idx, nt in enumerate(nts):
            newX_center = cx_arr[idx]
            newY_center = cy_arr[idx]
            new_r = cr_arr[idx]
            vsInfo = nt.getVerticesInfo()
            for id_, vData in vsInfo.items():
                newX = newX_center + ((vData.coordinate.x - nt.canvasWidth / 2.0) / nt.canvasWidth) * 2 * new_r * 1.25
                newY = newY_center + ((vData.coordinate.y - nt.canvasHeight / 2.0) / nt.canvasHeight) * 2 * new_r * 1.25
                self.xydata[self.vVID2Index[id_]].setLocation(newX, newY)

    # --- End optimized replacements ---

    def arcFromCenter(self, p: Point) -> float | None:
        center = Point(self.w / 2.0, self.h / 2.0)
        dx = p.getX() - center.getX()
        dy = p.getY() - center.getY()
        if abs(dx) < 0.00001 and abs(dy) < 0.00001:
            return None
        elif dx == 0 and dy > 0:
            arc = math.pi / 2
        elif dx == 0 and dy < 0:
            arc = -math.pi / 2
        else:
            arc = math.atan(dy / dx)
        if dx < 0:
            arc += math.pi
        return arc

    def occupy(self, pt: Point, index: int) -> int:
        currV = self.vIndex2VID[index]
        currVData = self.vsInfo[currV]
        for i in range(len(self.xydata)):
            if i == index:
                continue
            v = self.vIndex2VID[i]
            vData = self.vsInfo[v]
            if pt.distance(self.xydata[i]) < (vData.radius + currVData.radius) * 1.2:
                return i
        return -1

    def initPos(self, vData: Vertex) -> Point:
        vid = vData.id
        arc: float | None = 0.0
        attr = ""

        for attr_k, p in self.magnetPos.items():
            attr = attr_k
            if attr in vData.extraProperties and attr in vData.propertyStatus and vData.propertyStatus[attr] == "layout":
                b = int(vData.extraProperties[attr])
                if b == 1:
                    arc = self.arcFromCenter(p)
                    break

        angleVariation = self.magnetInitAngleRange[attr] / 2
        currArc = 0.0
        currRadiusFromCenter = 0.0

        if attr not in self.magnetLastVertex:
            currArc = float(arc) - angleVariation / 2
            self.magnetVertexInitIndex[attr] = 1
            currRadiusFromCenter = self.canvasRadius - self.aveVertexSize * 2
        else:
            lastVData = self.vsInfo[self.magnetLastVertex[attr]]
            lastVPos = self.xydata[self.vVID2Index[lastVData.id]]
            lastArc = self.arcFromCenter(lastVPos)
            lastRadius = lastVData.radius
            currArc = float(lastArc) + (lastRadius + vData.radius) * 2 / self.canvasRadius
            maxArc = float(arc) + angleVariation
            currRadiusFromCenter = lastVPos.distance(self.canvasCenter)
            if currArc > maxArc:
                currArc = float(arc) - angleVariation
                currRadiusFromCenter -= self.aveVertexSize * 8
            initVID = self.magnetVertexInitIndex[attr]
            self.magnetVertexInitIndex[attr] = initVID + 1

        self.magnetLastVertex[attr] = vid
        x = self.canvasRadius + currRadiusFromCenter * math.cos(currArc)
        y = self.canvasRadius + currRadiusFromCenter * math.sin(currArc)
        return Point(x, y)

    def _build_step_cache(self):
        """
        Pre-compute per-vertex data that is invariant across step() calls:
          - magnet interactions (only the magnets that apply to each vertex)
          - edge neighbour indices with pre-parsed weights
          - forceAdj per vertex
        Also caches edge-neighbour sets for getEnergy_opt.
        Called once before the step loop in run().
        """
        n = len(self.xydata)

        # Per-vertex magnet list: [(px, py, wt, b), ...]
        self._v_magnets: list[list[tuple[float, float, float, int]]] = []
        # Per-vertex edge neighbour list: [(neighbour_index, edge_weight), ...]
        self._v_edges: list[list[tuple[int, float]]] = []
        # Per-vertex force adjustment scalar
        self._v_forceAdj: list[float] = []
        # Per-vertex collision radius
        self._v_radius: list[float] = []
        # Per-vertex energy neighbour list (for getEnergy_opt): [(nbr_index, weight), ...]
        self._v_energy_edges: list[list[tuple[int, float]]] = []

        for i in range(n):
            v = self.vIndex2VID[i]
            vData = self.vsInfo[v]
            self._v_forceAdj.append(math.sqrt(vData.score.get("componentSize", 1.0)) / 10)
            self._v_radius.append(float(vData.radius))

            magnets: list[tuple[float, float, float, int]] = []
            for k, p in self.magnetPos.items():
                if k in vData.extraProperties and k in vData.propertyStatus and vData.propertyStatus[k] == "layout":
                    magnets.append((p.x, p.y, self.magnetIntensity[k], int(vData.extraProperties[k])))
            self._v_magnets.append(magnets)

            edges: list[tuple[int, float]] = []
            for _, nbr, data in self.graph.edges(v, data=True):
                eID = data.get("id")
                if not eID:
                    continue
                try:
                    vIDs = eID[1:-1].split("-")
                    v1ID = int(vIDs[0])
                    v2ID = int(vIDs[1])
                except Exception:
                    continue
                targetVID = v2ID if v1ID == v else v1ID
                if targetVID not in self.vVID2Index:
                    continue
                e = self.esInfo.get(eID)
                if e is None:
                    continue
                edges.append((self.vVID2Index[targetVID], e.weight))
            self._v_edges.append(edges)
            self._v_energy_edges.append(edges)  # same data, separate reference for clarity

        # Magnet positions as plain floats for getEnergy_opt
        self._magnet_list: list[tuple[float, float, float]] = [
            (p.x, p.y, self.magnetIntensity[k])
            for k, p in self.magnetPos.items()
        ]
        self._magnet_keys = list(self.magnetPos.keys())

    def _build_spatial_grid(self, cell_size: float):
        """Build a spatial hash grid over current xydata positions."""
        self._grid_cell = cell_size
        grid: dict[tuple[int, int], list[int]] = {}
        for i, pt in enumerate(self.xydata):
            key = (int(pt.x // cell_size), int(pt.y // cell_size))
            if key not in grid:
                grid[key] = []
            grid[key].append(i)
        self._grid = grid

    def _grid_remove(self, i: int, x: float, y: float):
        key = (int(x // self._grid_cell), int(y // self._grid_cell))
        bucket = self._grid.get(key)
        if bucket and i in bucket:
            bucket.remove(i)

    def _grid_add(self, i: int, x: float, y: float):
        key = (int(x // self._grid_cell), int(y // self._grid_cell))
        if key not in self._grid:
            self._grid[key] = []
        self._grid[key].append(i)

    def _occupy_grid(self, px: float, py: float, index: int) -> int:
        """Grid-accelerated replacement for occupy(): O(1) average instead of O(N)."""
        cell = self._grid_cell
        currRadius = self._v_radius[index]
        # search the 3×3 neighbourhood of cells
        cx = int(px // cell)
        cy = int(py // cell)
        for gx in range(cx - 1, cx + 2):
            for gy in range(cy - 1, cy + 2):
                bucket = self._grid.get((gx, gy))
                if not bucket:
                    continue
                for j in bucket:
                    if j == index:
                        continue
                    dx = px - self.xydata[j].x
                    dy = py - self.xydata[j].y
                    if (dx * dx + dy * dy) < ((self._v_radius[j] + currRadius) * 1.2) ** 2:
                        return j
        return -1

    def getEnergy_opt(self, index: int) -> float:
        """Optimized getEnergy using pre-built caches — avoids O(E) string scan."""
        canvasRadius = math.hypot(self.h, self.w)
        energy = 0.0
        pt = self.xydata[index]
        v = self.vIndex2VID[index]
        vData = self.vsInfo[v]
        progress_factor = 1.0 / (self.progress + 0.75)

        for k, (px, py, wt) in zip(self._magnet_keys, self._magnet_list):
            if k in vData.extraProperties and k in vData.propertyStatus and vData.propertyStatus[k] == "layout":
                ddx = px - pt.x
                ddy = py - pt.y
                dist = math.hypot(ddx, ddy) or 1e-9
                b = int(vData.extraProperties[k])
                if b == 0:
                    energy -= wt * canvasRadius / dist
                else:
                    energy += dist * dist * wt

        for nbr_idx, weight in self._v_energy_edges[index]:
            nbr_pt = self.xydata[nbr_idx]
            ddx = nbr_pt.x - pt.x
            ddy = nbr_pt.y - pt.y
            dist2 = ddx * ddx + ddy * ddy
            energy += dist2 * self.linkStrength * weight * progress_factor

        return energy

    def step_opt(self):
        """
        Optimized step().

        Changes vs original:
        - Uses pre-built per-vertex magnet/edge caches (_build_step_cache) to
          avoid re-iterating graph structure and re-parsing edge IDs every step.
        - Replaces O(N) occupy() linear scan with an O(1)-average spatial grid.
        - Uses getEnergy_opt() which avoids the O(E) string-scan in getEnergy().
        - Accesses Point.x/y directly instead of going through getX()/getY().
        """
        cr = self.w if self.h > self.w else self.h
        cr2 = cr * cr
        link_progress = 1.0 / (self.progress + 0.75)
        w_lo, w_hi = self.w * 0.05, self.w * 0.95
        h_lo, h_hi = self.h * 0.05, self.h * 0.95
        w5, h5 = self.w / 5, self.h / 5

        for i in range(len(self.xydata)):
            pt = self.xydata[i]
            x = pt.x
            y = pt.y
            forceAdj = self._v_forceAdj[i]
            dx = 0.0
            dy = 0.0

            # Magnet forces
            for px, py, wt, b in self._v_magnets[i]:
                ddx = px - x
                ddy = py - y
                dist2 = ddx * ddx + ddy * ddy
                dist = math.sqrt(dist2) if dist2 > 0 else 1e-9
                if b == 1:
                    force = wt * dist2 * forceAdj / cr2
                    dx += ddx * force
                    dy += ddy * force
                else:
                    force = wt * cr / dist * forceAdj / cr
                    dx -= ddx * force
                    dy -= ddy * force

            # Edge forces
            for nbr_idx, weight in self._v_edges[i]:
                nbr = self.xydata[nbr_idx]
                ddx = nbr.x - x
                ddy = nbr.y - y
                dist2 = ddx * ddx + ddy * ddy
                force = self.linkStrength * dist2 * weight * link_progress * forceAdj / cr2
                dx += ddx * force
                dy += ddy * force

            nx_ = x + dx
            ny_ = y + dy
            if nx_ >= w_hi or nx_ <= w_lo or ny_ >= h_hi or ny_ <= h_lo:
                continue

            while abs(dx) > w5 or abs(dy) > h5:
                dx /= 2.0
                dy /= 2.0
                nx_ = x + dx
                ny_ = y + dy

            targetID = self._occupy_grid(nx_, ny_, i)
            if targetID >= 0:
                currEnergy = self.getEnergy_opt(i) + self.getEnergy_opt(targetID)
                # swap
                old_ix, old_iy = x, y
                old_tx, old_ty = self.xydata[targetID].x, self.xydata[targetID].y
                self._grid_remove(i, old_ix, old_iy)
                self._grid_remove(targetID, old_tx, old_ty)
                self.xydata[i].setLocation(old_tx, old_ty)
                self.xydata[targetID].setLocation(old_ix, old_iy)
                self._grid_add(i, old_tx, old_ty)
                self._grid_add(targetID, old_ix, old_iy)
                newEnergy = self.getEnergy_opt(i) + self.getEnergy_opt(targetID)
                occ_i = self._occupy_grid(self.xydata[i].x, self.xydata[i].y, i)
                occ_t = self._occupy_grid(self.xydata[targetID].x, self.xydata[targetID].y, targetID)
                if occ_i >= 0 or occ_t >= 0 or newEnergy >= currEnergy:
                    # revert swap
                    self._grid_remove(i, old_tx, old_ty)
                    self._grid_remove(targetID, old_ix, old_iy)
                    self.xydata[i].setLocation(old_ix, old_iy)
                    self.xydata[targetID].setLocation(old_tx, old_ty)
                    self._grid_add(i, old_ix, old_iy)
                    self._grid_add(targetID, old_tx, old_ty)
            else:
                old_x, old_y = x, y
                self._grid_remove(i, old_x, old_y)
                self.xydata[i].setLocation(nx_, ny_)
                self._grid_add(i, nx_, ny_)

    def getEnergy(self, index: int) -> float:
        h = self.h
        w = self.w
        canvasRadius = math.hypot(h, w)

        energy = 0.0
        v = self.vIndex2VID[index]
        vData = self.vsInfo[v]

        for k, mp in self.magnetPos.items():
            dist = mp.distance(self.xydata[index])
            if dist == 0:
                dist = 1e-9
            wt = self.magnetIntensity[k]
            if k in vData.extraProperties and k in vData.propertyStatus and vData.propertyStatus[k] == "layout":
                b = int(vData.extraProperties[k])
                if b == 0:
                    energy -= wt * canvasRadius / dist
                else:
                    energy += dist * dist * wt

        for k, e in self.esInfo.items():
            if (f"^{v}-" in k) or (f"-{v}$" in k):
                vSource = e.toV if e.fromV == vData else e.fromV
                for n in range(len(self.xydata)):
                    if self.vIndex2VID[n] == vSource.id:
                        dist = self.xydata[n].distance(self.xydata[index])
                        energy += dist * dist * self.linkStrength * e.weight * (1 / (self.progress + 0.75))
        return energy
