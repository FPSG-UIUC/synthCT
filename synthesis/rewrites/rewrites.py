# Implements AST rewrite rules to simplify complex K-opcodes
# aka. Node-splitting

from collections import defaultdict

from loguru import logger
import networkx as nx

from semantics.typing import Typing


class Rewriter:
    def __init__(self, rules="all"):
        if rules == "all":
            rules = REWRITE_RULES.keys()
            self.rules = []
            for rule in rules:
                self.rules.append(REWRITE_RULES[rule](rule))

    def rewrite_inst(self, inst):
        # TODO: Have an option to make a copy of AST instead of replacement
        for k, sem in inst.sems.items():
            self.rewrite_ast(sem)
        inst.semantics_to_svg("DIVB-R8")

    def rewrite_ast(self, ast):
        for rule in self.rules:
            nodes = list(ast.sem.nodes())
            for node in nodes:
                rule.rewrite_node(node, ast)


class RewriteRule:
    def __init__(self, name):
        self.rule_name = name

    def can_rewrite_node(self, opcode):
        raise NotImplementedError

    def rewrite_node(self, node, ast):
        raise NotImplementedError


class RewriteRotate(RewriteRule):
    def __init__(self, name):
        super().__init__(name)

    def can_rewrite_node(self, opcode):
        return opcode in ["rol"]

    def rewrite_node(self, node, sem):
        opcode = sem.node_data(node, "data")
        if not self.can_rewrite_node(opcode):
            return
        """
        assume %1 is already mod-ed
        rol %0 %1 ->
          (or (lshr %0 (sub %w %1))
              (shl %0 %1))
        """
        op1, op2 = sem.successors_ordered(node)
        width = sem.node_data(op1, Typing.KEY).sz

        subn = sem.add_node(data="subMInt")
        shln = sem.add_node(data="shiftLeftMInt")
        lshrn = sem.add_node(data="lshrMInt")
        orn = sem.add_node(data="orMInt")

        width_const = sem.add_node(data="mi")
        widthi = sem.add_node(data=width)
        width_w = sem.add_node(data=width)

        sem.add_edge(orn, lshrn, idx=0)
        sem.add_edge(orn, shln, idx=1)

        sem.add_edge(lshrn, op1, idx=0)
        sem.add_edge(lshrn, subn, idx=1)

        sem.add_edge(shln, op1, idx=0)
        sem.add_edge(shln, op2, idx=1)

        sem.add_edge(width_const, width_w, idx=0)
        sem.add_edge(width_const, widthi, idx=1)

        sem.add_edge(subn, width_const, idx=0)
        sem.add_edge(subn, op2, idx=1)

        sem.redirect_incoming(node, orn)
        sem.remove_node(node)


class RewriteDiv(RewriteRule):
    def __init__(self, name):
        super().__init__(name)

        # Some state to use while constructing
        self.keys = defaultdict(None)
        self.graph = None
        self.op_width = None

    def can_rewrite_node(self, opcode):
        if not isinstance(opcode, str):
            return False
        return opcode.startswith("div_quotient_") or opcode.startswith("div_remainder_")

    def _add_token(self, token):
        data = token
        try:
            value = int(token)
            data = value
        except ValueError:
            if token in self.keys:
                return self.keys[token]

        return self.graph.add_node(data=data)

    def _process_subtree(self, stack):
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
            self.graph.sem.add_edge(opcode, op, idx=idx)

        return opcode

    def _add_sexp_to_graph(self, expression):
        # logger.debug(f"Trying to add expression: {expression}")
        stack = []
        ctoken = ""
        # Invariant: stack should only contain node indices, '('
        for token in expression:
            if token == "(":
                stack.append(token)
            elif token == ")":
                if ctoken:
                    ni = self._add_token(ctoken)
                    stack.append(ni)
                    ctoken = ""
                ni = self._process_subtree(stack)
                stack.append(ni)
            elif token == " ":
                if not ctoken:
                    continue
                # TODO: Add token to graph and push a node index
                ni = self._add_token(ctoken)
                stack.append(ni)
                ctoken = ""
            else:
                ctoken += token

        assert len(stack) == 1, "Incomplete parsing?"
        return stack[0]

    def _compute_one_iteration(self, iter):
        # R' = (bvor (bvshl R (1 16)) (bvand (bvlshr dividend (bv i 16)) (bv 1 16))))
        # R'' = (ite (bvuge R' divisor)
        #   (bvsub R' divisor)
        #   (R'))
        # Q' = (ite (bvuge R' divisor)
        #   (bvor Q (bvshl (bv 1 16) (bv iter 16)))
        #   (Q))
        # -----------------------------------------------------------------
        # Q = Q'
        # R = R''
        w = self.op_width
        rp_sexpr = (
            f"(orMInt (shiftLeftMInt R (mi {w} 1))"
            f"        (andMInt (lshrMInt dividend (mi {w} {iter}))"
            f"                 (mi {w} 1)))"
        )
        rp_idx = self._add_sexp_to_graph(rp_sexpr)
        self.keys["R'"] = rp_idx

        rpp_sexpr = (
            "(#ifMInt (ugeMInt R' divisor)"
            "         (subMInt R' divisor)"
            "         R')"
        )
        rpp_idx = self._add_sexp_to_graph(rpp_sexpr)
        self.keys["R''"] = rpp_idx

        qp_sexpr = (
            "(#ifMInt (ugeMInt R' divisor)"
            f"        (orMInt Q (shiftLeftMInt (mi {w} 1) (mi {w} {iter})))"
            f"        Q)"
        )
        qp_idx = self._add_sexp_to_graph(qp_sexpr)
        self.keys["Q'"] = qp_idx

        self.keys["Q"] = self.keys["Q'"]
        self.keys["R"] = self.keys["R''"]

    def _do_init(self, w, node):
        # Div and remainder are both expected to have two operands to `node`
        # Assumption: both op1 and op2 are correct bitwidths already in the original
        # semantics
        op1, op2 = self.graph.successors_ordered(node)

        ty1 = self.graph.node_data(op1, Typing.KEY).sz
        ty2 = self.graph.node_data(op2, Typing.KEY).sz

        # Need this only if type checking does not pass
        if ty2 < ty1:
            op2 = self.graph.add_zext(op2, ty1)

        self.op_width = ty1

        self.keys["dividend"] = op1
        self.keys["divisor"] = op2

        init_zero = f"(mi {ty1} 0)"
        q_i = self._add_sexp_to_graph(init_zero)
        r_i = self._add_sexp_to_graph(init_zero)

        self.keys["Q"] = q_i
        self.keys["R"] = r_i

    def _do_finish(self, w, node):
        outwidth = w
        node_type = self.graph.node_data(node, "data")

        op1, op2 = self.graph.successors_ordered(node)
        ty1 = self.graph.node_data(op1, Typing.KEY).sz

        output = None

        if node_type.startswith("div_quotient_"):
            output = self.keys["Q"]
        elif node_type.startswith("div_remainder_"):
            output = self.keys["R"]

        if outwidth < ty1:
            output = self.graph.add_extract(output, outwidth - 1, 0)

        self.graph.redirect_incoming(node, output)
        self.graph.remove_node(node)

    def _do_cleanup(self):
        wl = list(self.graph.sem.nodes())
        while wl:
            current = wl.pop(0)
            preds = list(self.graph.predecessors(current))
            if not preds and current != self.root:
                nexts = list(self.graph.successors(current))
                self.graph.remove_node(current)
                for n in nexts:
                    if n not in wl:
                        wl.append(n)

    def rewrite_node(self, node, sem):
        self.graph = sem
        self.root = sem.root

        node_data = self.graph.node_data(node, "data")
        if not self.can_rewrite_node(node_data):
            return

        _, width = node_data.rsplit("int", 1)
        width = int(width)
        self._do_init(width, node)
        # for iter in range(self.op_width - 1, -1, -1):
        for iter in range(1, -1, -1):
            self._compute_one_iteration(iter)

        self._do_finish(width, node)
        self._do_cleanup()


REWRITE_RULES = {
    "rewrite_rotate": RewriteRotate,
    "rewrite_divide": RewriteDiv,
}
