# Implements methods to generate embedding for semantics
#
# The current implementations are purely experimental to gain more insights
# into the graph structures of the semantics

from loguru import logger

from karateclub import Graph2Vec, GL2Vec, FGSD
from karateclub.graph_embedding.feathergraph import FeatherGraph
import networkx as nx

from semantics.typing import Typing


class Embedding:
    def __init__(self, keys, sems):
        self.keys = []
        self.gs = []
        self.embeddings = []

        for key, sem in zip(keys, sems):
            success = self._copy_graph(sem)
            if success:
                self.keys.append(key)
            else:
                logger.warning(f"Skipped: {key}")

    def _copy_graph(self, g):
        if not g.to_undirected():
            return False

        newg = nx.Graph()
        mapping = {}

        for idx, node in enumerate(g.nodes()):
            mapping[node] = idx
            label = g.nodes[node]["data"]
            node_type = g.nodes[node][Typing.KEY]
            newg.add_node(idx, feature=f"({label}:{node_type})")

        for src, dst in g.edges():
            newg.add_edge(mapping[src], mapping[dst])

        self.gs.append(newg)

        return True

    @staticmethod
    def embed(keys, sems):
        """
        Top-level method that takes a list of semantics and generates graph
        embeddings

        args:
        sem: list(`semantics.container.Semantics`)
        """
        embed = Embedding(keys, sems)
        embed.do_embed()

        return embed

    def do_embed(self):
        # Set the key "feature"
        self.model = Graph2Vec(
            epochs=30, wl_iterations=3, attributed=True, dimensions=128
        )
        self.model.fit(self.gs)
        self.embeddings = self.model.get_embedding()

    def get_embedding(self, key):
        idx = self.keys.index(key)
        return self.embeddings[idx]

    def get_all(self):
        for key, emb in zip(self.keys, self.embeddings):
            yield key, emb
