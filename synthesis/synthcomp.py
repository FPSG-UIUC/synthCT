# Implements strategies for choosing components for synthesis based on the
# specification to synthesize

from collections import defaultdict
import random

import networkx as nx

from loguru import logger
from sklearn.neighbors import NearestNeighbors

from learning.embedding import Embedding
from semantics.typing import Typing


class CompSelector:
    def __init__(self, isa):
        pass

    def components_for(self, inst, count=32):
        raise NotImplementedError


class SimpleSelector(CompSelector):
    def __init__(self, isa):
        self.sems = isa

    def components_for(self, inst, count=32):
        components = []
        for comp in self.sems:
            # Ensure spec/inst is not in components
            if comp.name == inst.name:
                continue
            components.append(comp)
        return components


class KNNSelector(CompSelector):
    def __init__(self, isa):
        self.sems = isa

        self.gs = []
        self.keys = []

        for instsem in self.sems:
            reg_graph = nx.DiGraph()
            for reg, sem in instsem.iter_output_regs():
                reg_graph = nx.union(reg_graph, sem.sem, rename=(None, f"{reg}-"))

            if reg_graph:
                self.gs.append(reg_graph)
                self.keys.append([sem.name, "REG"])

            flag_graph = nx.DiGraph()
            for flag, sem in instsem.iter_output_flags():
                flag_graph = nx.union(flag_graph, sem.sem, rename=(None, f"{flag}-"))

            if flag_graph:
                self.gs.append(flag_graph)
                self.keys.append([sem.name, "FLAG"])

        self.eb = Embedding.embed(self.keys, self.gs)
        self.nn = NearestNeighbors()
        self.nn.fit([y for _, y in self.eb.get_all()])

    def components_for(self, inst, count=32, mode="REG"):
        # Check if there are enough components
        if count > len(self.keys):
            logger.warning(
                f"Requested for more subcomponents than available: "
                f"Choosing {len(self.keys)} instead of {count}"
            )
            count = len(self.keys)

        # Here, we get some number > count to account for the same
        # instruction being chosen multiple times due to multiple output
        # registers
        knn = None

        #max_nodes = None
        #for reg, sem in inst.iter_output_regs():
            #if not max_nodes:
                #max_nodes = reg
            #else:
                #cur = len(list(sem.sem.nodes()))
                #maxn = len(list(inst.sems[max_nodes].sem.nodes()))
                #if cur > maxn:
                    #max_nodes = reg

        #if max_nodes is None:
            #logger.warning(f"No output registers for {inst.name}?")
            #for reg, sem in inst.iter_output_flags():
                #if not max_nodes:
                    #max_nodes = reg
                #else:
                    #cur = len(list(sem.sem.nodes()))
                    #maxn = len(list(inst.sems[max_nodes].sem.nodes()))
                    #if cur > maxn:
                        #max_nodes = reg
            #if max_nodes is None:
                #logger.warning(f"No output flags for {inst.name}?")
                #return []

        key = [inst.name, mode]
        knn = self.nn.kneighbors([self.eb.get_embedding(key)], (count + 1) * 6)

        assert knn is not None, "No output registers?"

        cnames = []
        for j, idx in enumerate(knn[1][0]):
            name, reg = self.eb.keys[idx]
            if name in cnames:
                continue
            # Do not select the synthesis spec instruction in the components
            if name == inst.name:
                continue

            cnames.append(name)
            # Check if we reached threshold number of keys
            if len(cnames) >= count:
                break

        assert len(cnames) == count, "Incorrect number of components chosen?"

        components = [None] * count
        for sem in self.sems:
            if sem.name in cnames:
                components[cnames.index(sem.name)] = sem

        return components


class JaccardSelector(CompSelector):
    def __init__(self, isa):
        self.sems = isa
        self.set_map = dict()
        self.sets = []

        self._compute_sets(isa)

    def _compute_sets(self, isa):
        for idx, inst in enumerate(isa):
            # Calculate the set for register outputs
            current = defaultdict(lambda: 0)
            for reg, sem in inst.iter_output_regs():
                for node in sem.sem.nodes():
                    nd = sem.node_data(node, "data")
                    nt = sem.node_data(node, Typing.KEY)
                    label = f"({nd}:{nt})"
                    current[label] += 1
            self.sets.append(current)
            self.set_map[(sem.name, "REG")] = 2 * idx

            # Calculate set for flag outputs
            current = defaultdict(lambda: 0)
            for flag, sem in inst.iter_output_flags():
                for node in sem.sem.nodes():
                    nd = sem.node_data(node, "data")
                    nt = sem.node_data(node, Typing.KEY)
                    label = f"({nd}:{nt})"
                    current[label] += 1
            self.sets.append(current)
            self.set_map[(sem.name, "FLAG")] = 2 * idx + 1

    def _compute_generalized_jaccard(self, idx1, idx2):
        set_1 = self.sets[idx1]
        set_2 = self.sets[idx2]

        all_keys = set(set_1.keys()).union(set(set_2.keys()))

        num = 0.0
        denom = 0.0

        for key in all_keys:
            v1 = set_1[key]
            v2 = set_2[key]
            num += min(v1, v2)
            denom += max(v1, v2)

        return float(num) / float(denom)

    def components_for(self, inst, count=32, mode="REG"):
        scores = dict()
        current = self.set_map[(inst.name, mode)]
        for k, v in self.set_map.items():
            if k[0] == inst.name:
                continue
            scores[k] = self._compute_generalized_jaccard(current, v)

        components = []
        selected = []
        for k, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True):
            idx = self.set_map[k] // 2
            sem = self.sems[idx]
            # Avoid duplicates
            if sem.name not in selected:
                components.append(self.sems[idx])
                selected.append(sem.name)

        if len(components) < count:
            logger.warning("Selected less than requested number of components!")
            return components

        return components[:count]


class RKNNSelector(KNNSelector):
    def __init__(self, isa):
        super().__init__(isa)

    def components_for(self, inst, count=18):
        components = super().components_for(inst, 30)
        # Randomly select count from returned components
        idxs = random.sample(range(0, len(components)), count)
        rcomps = []
        for idx in idxs:
            rcomps.append(components[idx])

        return rcomps


class StaticSelector(CompSelector):
    def __init__(self, isa):
        self.isa = isa

        self.static = {
            "ROLQ-R64-ONE": [
                "CMPXCHGQ-R64-R64",
                "XORW-R16-R16",
                "INCQ-R64",
                "DECQ-R64",
                "XCHGB-R8-R8",
                "XCHGQ-R64-R64",
                "ROLQ-R64-CL",
                "MOVSBL-R32-R8",
                "ANDQ-R64-R64",
                "ORL-R32-R32",
                "BLSRQ-R64-R64",
                "NEGQ-R64",
                "XORL-R32-R32",
                "MOVSLQ-R64-R32",
                "MOVSBL-R32-RH",
            ],
            "SALQ-R64-CL": [
                "SHLQ-R64-CL",
                "SETPO-R8",
                "CMOVPOL-R32-R32",
                "SETNS-R8",
                "SBBQ-R64-R64",
                "CMPXCHGB-RH-RH",
                "SETNO-R8",
                "SARW-R16-CL",
                "IMULW-R16-R16",
                "SHRW-R16-CL",
                "SUBQ-R64-R64",
                "CMOVBL-R32-R32",
                "SHRQ-R64-CL",
                "SARQ-R64-ONE",
                "CMOVNPL-R32-R32",
                "SHLW-R16-CL",
                "RCRQ-R64-ONE",
                "SETNP-R8",
            ],
            "ROLQ-R64-CL": [
                "MOVQ-R64-R64",
                "ANDQ-R64-R64",
                "SUBQ-R64-R64",
                "SHRQ-R64-R64",
                "SHLQ-R64-R64",
                "ORQ-R64-R64",
            ],
        }

    def components_for(self, inst, count=None):
        comps = []
        for sem in self.isa:
            if sem.name in self.static[inst.name]:
                comps.append(sem)

        return comps


selector = {
    "simple": SimpleSelector,
    "knn": KNNSelector,
    "rknn": RKNNSelector,
    "jaccard": JaccardSelector,
    "static": StaticSelector,
}
