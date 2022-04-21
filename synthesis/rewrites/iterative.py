# Implements iterative rewrites for node splitting complex K-AST nodes

from loguru import logger

from semantics.containers import InstructionSemantics, Semantics, Operand
from semantics.typing import Typing


class IterativeRewriteDivide:
    def __init__(self, sem):
        self.main = sem

        self.keys = dict()

        # TODO: Set based on real types
        self.w = 8
        self.results = dict()

        #return self.create_splits()

    @property
    def name(self):
        return self.main.name

    @classmethod
    def can_rewrite(cls, inst):
        for k, sem in inst.sems.items():
            for node in sem.sem.nodes():
                if cls.can_rewrite_node(sem.node_data(node, key='data')):
                    return True
        return False

    @classmethod
    def can_rewrite_node(cls, opcode):
        logger.debug(opcode)
        if not isinstance(opcode, str):
            return False
        return opcode.startswith("div_quotient_") or opcode.startswith("div_remainder_")


    def _add_token(self, graph, keys, token):
        data = token
        try:
            value = int(token)
            data = value
        except ValueError:
            if token in keys:
                idx = keys[token]
                preds = list(graph.predecessors(idx))
                if preds:
                    return graph.clone_subtree(idx)
                else:
                    return keys[token]

        return graph.add_node(data=data)

    def _process_subtree(self, graph, stack):
        ops = []
        while stack[-1] != "(":
            ops.append(stack.pop(-1))

        stack.pop(-1)

        ops = ops[::-1]

        opcode = ops[0]
        operands = []
        if len(ops) > 1:
            operands = ops[1:]

        for idx, op in enumerate(operands):
            graph.sem.add_edge(opcode, op, idx=idx)

        return opcode

    def _add_sexp_to_graph(self, graph, keys, expression):
        # logger.debug(f"Trying to add expression: {expression}")
        stack = []
        ctoken = ""
        # Invariant: stack should only contain node indices, '('
        for token in expression:
            if token == "(":
                stack.append(token)
            elif token == ")":
                if ctoken:
                    ni = self._add_token(graph, keys, ctoken)
                    stack.append(ni)
                    ctoken = ""
                ni = self._process_subtree(graph, stack)
                stack.append(ni)
            elif token == " ":
                if not ctoken:
                    continue
                # TODO: Add token to graph and push a node index
                ni = self._add_token(graph, keys, ctoken)
                stack.append(ni)
                ctoken = ""
            else:
                ctoken += token

        assert len(stack) == 1, "Incomplete parsing?"
        return stack[0]

    def _create_init(self):
        logger.debug("Creating init for div")

        ty = "Gpr"
        operands = [
            Operand("R1", 64, ty),
            Operand("R2", 64, ty),
        ]

        init_inst = InstructionSemantics("div_init", "div_init", operands)
        semq = Semantics("R1")
        semr = Semantics("R2")

        nq = self._add_sexp_to_graph(semq, {}, f"(mi {self.w} 0)")
        nr = self._add_sexp_to_graph(semr, {}, f"(mi {self.w} 0)")

        if self.w < 64:
            semq.add_zext(nq, 64)
            semr.add_zext(nr, 64)

        init_inst.define_semantics_for("R1", semq)
        init_inst.define_semantics_for("R2", semr)

        Typing.infer(semq)
        Typing.infer(semr)

        init_inst.set_metadata("main_task", self.name)

        return init_inst

    def _create_loop(self):
        logger.debug("Creating loop for div")

        ty = "Gpr"
        operands = [
            Operand("R1", 64, ty),  # dividend
            Operand("R2", 64, ty),  # divisor
            Operand("R3", 64, ty),  # Q
            Operand("R4", 64, ty),  # R
            Operand("R5", 64, ty),  # iter
        ]

        loop_inst = InstructionSemantics("div_loop", "div_loop", operands)
        semq = Semantics("R3")
        semr = Semantics("R4")

        qkeys = {}
        rkeys = {}

        # Add all the operands in
        for op in operands:
            idx = semq.add_node(data=op.name)
            qkeys[op.name] = idx

            if op.name == "R3":
                continue

            idx = semr.add_node(data=op.name)
            rkeys[op.name] = idx

        w = 64

        rp_sexpr = (
            f"(orMInt (shiftLeftMInt R4 (mi {w} 1))"
            f"        (andMInt (lshrMInt R1 R5)"
            f"                 (mi {w} 1)))"
        )

        rpp_sexpr = (
            "(#ifMInt (ugeMInt R' R2)"
            "         (subMInt R' R2)"
            "         R')"
        )

        qp_sexpr = (
            "(#ifMInt (ugeMInt R' R2)"
            f"        (orMInt R3 (shiftLeftMInt (mi {w} 1) R5))"
            f"        R2)"
        )

        rp_idx = self._add_sexp_to_graph(semq, qkeys, rp_sexpr)
        qkeys["R'"] = rp_idx
        logger.debug(f"R': {rp_idx}")
        #_ = self._add_sexp_to_graph(semq, qkeys, rpp_sexpr)
        _ = self._add_sexp_to_graph(semq, qkeys, qp_sexpr)

        # Compute remainder
        rp_idx = self._add_sexp_to_graph(semr, rkeys, rp_sexpr)
        rkeys["R'"] = rp_idx
        _ = self._add_sexp_to_graph(semr, rkeys, rpp_sexpr)

        loop_inst.define_semantics_for("R3", semq)
        loop_inst.define_semantics_for("R4", semr)

        Typing.infer(semq)
        Typing.infer(semr)

        loop_inst.set_metadata("main_task", self.name)
        return loop_inst

    def _create_finalize(self):
        pass

    def create_splits(self):
        return [#self._create_init(),
                self._create_loop()]

    def on_success(self, st, result):
        pass

    def is_done(self):
        pass

    def stitch(self):
        pass
