# Defines some classes to be used to store the results of parsing
import networkx as nx
import tempfile as tf
import copy
from collections import defaultdict
import hashlib

from loguru import logger

import claripy

import semantics.mappings as maps
from semantics.typing import Typing, BitVec, Bool


class InstructionSemantics:
    def __init__(self, name, opcode, operands):
        self.name = name
        self.opcode = opcode
        self.operands = operands
        self.sems = dict()
        self.alias = []

        self.instruction_metadata = dict()

    def is_pseudo(self):
        return False

    def set_metadata(self, key, value):
        self.instruction_metadata[key] = value

    def get_metadata(self, key, default=None):
        return self.instruction_metadata.get(key, default)

    def define_semantics_for(self, key, sem):
        self.sems[key] = sem
        sem._instsem = self

    def undef_semantics_for(self, key):
        del self.sems[key]

    def add_alias(self, opcode, operands):
        self.alias.append({"opcode": opcode, "operands": operands})

    @property
    def is_float(self):
        return self.instruction_metadata.get("float", False)

    @property
    def is_vec(self):
        return self.instruction_metadata.get("vec", False)

    def is_flag_reg(self, reg):
        # XXX: This is a fragile heuristic and must be generalized if/when we
        # generalize to other/more architectures, perhaps by looking at the
        # flag register information for the specific architecture
        # This works fine for x86 as no real register other than the flags
        # ends with an "F"
        if reg[-1] == "F":
            return True
        return False

    # Iterators
    # Some methods to iterate over the outputs defined in the semantics

    def iter_semantics(self):
        return iter(self.sems)

    def iter_output_regs(self):
        for k, sem in self.sems.items():
            if not self.is_flag_reg(k):
                yield k, sem

    def iter_outputs(self):
        for k, sem in self.sems.items():
            yield k, sem

    def iter_output_flags(self):
        for k, sem in self.sems.items():
            if self.is_flag_reg(k):
                yield k, sem

    def clone_with_keys(self, keys, rename='auto'):
        new_inst = copy.deepcopy(self)
        remove_keys = [
            k for k in new_inst.sems.keys() if k not in keys
        ]

        for rk in remove_keys:
            del new_inst.sems[rk]

        new_name = None
        if rename == 'auto':
            new_name = f"pseudo-{new_inst.name}-{'-'.join(keys)}"
        elif rename:
            new_name = rename

        if new_name:
            new_inst.name = new_name

        return new_inst

    def generate_hash(self):
        hash = []
        for k, sem in sorted(self.sems.items(), key=lambda x: x[0]):
            hash.append(f"{k}={sem.hash()}")
        hash = ";".join(hash)
        m = hashlib.md5()
        m.update(hash.encode("utf-8"))
        return m.hexdigest()

    def split_at_nidx(self, reg, nidx):
        # Split an instruction I, into I' and I'_r (the residual) at node
        # index specified by nidx.
        new_inst = copy.deepcopy(self)
        new_inst_r = copy.deepcopy(self)

        # Delete all keys other than "reg" in both the new split up
        # instructions
        for k in list(new_inst.sems):
            if k != reg:
                del new_inst.sems[k]
                del new_inst_r.sems[k]

        sem = self.sems[reg]

        #logger.debug(f"Splitting {self.name} at: {nidx}")
        #fname = f"./{self.name}_{reg}.svg"
        #logger.debug(f"Writing sem: {fname}!")
        #sem.semantics_to_svg(fname)

        new_sem, new_sem_r = sem.split_at_nidx(nidx)

        new_sem._instsem = new_inst
        new_sem_r._instsem = new_inst_r

        new_inst.sems[reg] = new_sem
        new_inst_r.sems[reg] = new_sem_r

        # XXX: This may accidently remove flag and harware registers!
        # XXX: Add output register "reg" into the operand list if it does not
        # already exist
        new_inst_r._fixup_metadata(reg, nidx, False)
        new_inst._fixup_metadata(reg, nidx, True)

        # Set metadata for parent instruction
        new_inst_r.set_metadata("parent", self.name)
        new_inst.set_metadata("parent", self.name)

        return new_inst, new_inst_r

    def _fixup_metadata(self, reg, nidx, split):
        hardware = ["RAX", "RDX", "RCX"]
        if nidx is not None:
            if split:
                self.name = f"{self.name}-{reg}-s{nidx}"
            else:
                self.name = f"{self.name}-{reg}-sr{nidx}"

            if not self.name.startswith("pseudo"):
                self.name = f"pseudo-{self.name}"

        sem = self.sems[reg]
        used = sem._used_regs_in_sem()

        self.operands = []

        add_reg = -1
        ridx = 0

        for name in used:
            if name in hardware:
                continue

            self.operands.append(Operand(name, 64, "R64"))
            if name == reg:
                add_reg = ridx
            ridx += 1

        # Ensure that the output operand is a part of the arguments and that
        # its the last operand.
        if reg in hardware:
            pass
        elif add_reg < 0:
            self.operands.append(Operand(reg, 64, "R64"))
        elif add_reg != len(self.operands) - 1:
            tmp = self.operands[-1]
            self.operands[-1] = self.operands[add_reg]
            self.operands[add_reg] = tmp

        Typing.infer(sem)

    # Debug/pretty-printing methods

    def __str__(self):
        return f"{self.name}: {self.opcode} {[str(x) for x in self.operands]}"

    def semantics_to_dot(self, prefix="./"):
        for key, sem in self.sems.items():
            fname = f"{prefix}{self.name}_{key}.dot"
            sem.semantics_to_dot(fname)

    def semantics_to_svg(self, prefix="./"):
        for key, sem in self.sems.items():
            fname = f"{prefix}{self.name}_{key}.svg"
            sem.semantics_to_svg(fname)

    def pretty_print_semantics(self):
        for k, sem in self.sems.items():
            print("[*] Semantics for:", k)
            sem.pretty_print_semantics()

    def pretty_print_prefix(self):
        for k, sem in self.sems.items():
            print("[*] Prefix Semantics for:", k)
            sem.pretty_print_prefix()

    def __deepcopy__(self, memo):
        res = type(self)(self.name, self.opcode, self.operands)
        res.sems = copy.deepcopy(self.sems, memo)

        # Fix up the ptr in sems
        for _r, sem in res.sems.items():
            sem._instsem = res

        res.alias = copy.deepcopy(self.alias, memo)
        res.instruction_metadata = copy.deepcopy(self.instruction_metadata)

        return res


class Operand:
    def __init__(self, name, sz, opty):
        self.name = name
        self.opty = opty
        self.sz = sz

    def __str__(self):
        return f"{self.name}:{self.opty}{self.sz}"


class ConstOperand:
    def __init__(self, value):
        self.value = value

    @property
    def name(self):
        return str(self)

    @property
    def sz(self):
        return "??"

    def __str__(self):
        return f"${self.value}"


class Semantics:
    def __init__(self, lhs):
        self.lhs = lhs
        self.sem = nx.DiGraph()
        self.mapping = dict()
        self.nidx = -1
        # NOTE: Readonly, do not mutate this object!
        self._instsem = None

    @property
    def instsem(self):
        return self._instsem

    @property
    def name(self):
        return self._instsem.name

    def hash(self):
        # TODO: cache this later on. Needs support for marking clean/dirty
        hash = self._pseudo_hash_subtree(self.root)
        hash = ";".join(hash)
        m = hashlib.md5()
        m.update(hash.encode("utf-8"))
        return m.hexdigest()

    def _pseudo_hash_subtree(self, node):
        hash = []
        nd = self.node_data(node, key="data")
        hash.append(f"{nd}")
        for succ in self.successors_ordered(node):
            h = self._pseudo_hash_subtree(succ)
            hash.extend(h)
        return hash

    def lookup_mapping(self, idx):
        return self.mapping[idx]["data"]

    # Basic graph manipulation API for semantics.
    def add_node(self, **data):
        idx = self._get_next_idx()
        self.sem.add_node(idx, **data)
        self.mapping[idx] = data
        return idx

    # Introduce ways to create some basic nodes
    # These functions just call into existing semantics/graph manipulation
    # functions and makes it easier to introduce new nodes into the graph

    def add_zext(self, opn, sz):
        tyn = self.add_node(data=sz)
        zn = self.add_node(data="zextMInt")
        self.add_edge(zn, opn, idx=0)
        self.add_edge(zn, tyn, idx=1)

        return zn

    def add_cvt_bool_to_bv(self, opn):
        cvtn = self.add_node(data="bool2bv")
        self.add_edge(cvtn, opn, idx=0)

        return cvtn

    def add_cvt_bv_to_bool(self, opn):
        cvtn = self.add_node(data="bv2bool")
        self.add_edge(cvtn, opn, idx=0)

        return cvtn

    def add_extract(self, bv, b, e):
        basen = self.add_node(data=b)
        endn = self.add_node(data=e)
        extractn = self.add_node(data="extractMInt")

        self.add_edge(extractn, bv, idx=0)
        self.add_edge(extractn, basen, idx=1)
        self.add_edge(extractn, endn, idx=2)

        return extractn

    def remove_node(self, nidx):
        self.sem.remove_node(nidx)

    def add_edge(self, source, target, idx=None):
        self.sem.add_edge(source, target, idx=idx)

    def add_operand_type_checked(self, source, target, idx):
        # TODO: Actually add code that fixes types
        self.sem.add_edge(source, target, idx=idx)

    def remove_edge(self, source, target):
        self.sem.remove_edge(source, target)

    def add_node_data(self, nidx, key, data):
        self.sem.nodes[nidx][key] = data

    def replace_node_data(self, nidx, data, key):
        self.sem.nodes[nidx][key] = data
        if key == "data":
            self.mapping[nidx][key] = data

    def redirect_incoming(self, fr, to):
        to_remove = []
        for pred in self.sem.predecessors(fr):
            idx = self.edge_data(pred, fr)["idx"]
            self.add_edge(pred, to, idx=idx)
            # To remove the other edge
            to_remove.append(pred)

        for p in to_remove:
            self.remove_edge(p, fr)

    def redirect_outgoing(self, fr, to):
        to_remove = []
        for succ in self.sem.successors(fr):
            idx = self.edge_data(fr, succ)["idx"]
            self.add_edge(to, succ, idx=idx)
            to_remove.append(succ)

        for s in to_remove:
            self.remove_edge(fr, succ)

    def edge_data(self, s, d, key=None):
        res = self.sem.get_edge_data(s, d)
        if key:
            res = res[key]
        return res

    def node_data(self, nidx, key=None, default=None):
        res = self.sem.nodes[nidx]
        if key:
            res = res.get(key, default)
        return res

    def successors(self, nidx):
        return self.sem.successors(nidx)

    def successors_ordered(self, nidx):
        unordered = list(self.successors(nidx))
        succs = [None] * len(unordered)
        for s in unordered:
            eidx = self.edge_data(nidx, s, "idx")
            succs[eidx] = s
        return succs

    def predecessors(self, nidx):
        return self.sem.predecessors(nidx)

    def iter_preorder(self):
        return nx.dfs_preorder_nodes(self.sem)

    def iter_postorder(self):
        return nx.dfs_postorder_nodes(self.sem)

    def subgraph(self, nodes):
        return self.sem.subgraph(nodes)

    def clone_subtree(self, nidx):
        subtree = nx.dfs_tree(self.sem, nidx).copy()
        copy_map = dict()

        for node in subtree.nodes():
            data = self.node_data(node, key='data')
            new_idx = self.add_node(data=data)
            copy_map[node] = new_idx

        for node in subtree.nodes():
            for idx, succ in enumerate(self.successors_ordered(node)):
                self.add_edge(
                    copy_map[node],
                    copy_map[succ],
                    idx=idx)

        return copy_map[nidx]

    def split_at_nidx(self, nidx):
        sem = self
        semg = self.sem

        new_sem_r = copy.deepcopy(self)
        new_sem = copy.deepcopy(self)

        new_sem_rg = copy.deepcopy(semg)
        new_semg = copy.deepcopy(semg)

        new_sem_r.sem = new_sem_rg
        new_sem.sem = new_semg

        new_semg.remove_nodes_from(
            [n for n in new_semg.nodes if n not in nx.dfs_tree(semg, nidx)]
        )

        # At the split point, the operands to the parent node need to be
        # "fixed"/repaired for the instruction to make sense.
        parents = list(sem.predecessors(nidx))

        # Create the residual graph: graph - new_semg
        newsem_root = new_sem_r.root
        for parent in parents:
            new_sem_rg.remove_edge(parent, nidx)
        new_sem_rg.remove_nodes_from(
            [
                n
                for n in new_sem_rg.nodes
                if n not in nx.dfs_tree(new_sem_rg, newsem_root)
            ]
        )

        # All operands used in the AST + the output register itself
        used = new_sem_r._used_regs_in_sem()

        logger.debug(f"Used: {used} Split@{nidx}")
        logger.debug(f"CURRENT ROOT: {self.root}")

        # Fixup types in both the split and the split residual instructions
        # Get the original type information for the node we're splitting at
        orig_ty = sem.node_data(nidx, key=Typing.KEY)

        # Replace the removed subtree in the residual with an available
        # abstract register
        available = [f"R{i}" for i in range(1, 8)]
        selected = None
        for regr in available:
            if regr not in used:
                used.append(regr)
                selected = new_sem_r.add_node(data=regr)
                new_sem_r.add_node_data(selected, "no_split", True)
                break

        assert selected is not None, "No available abstract register?"

        # In the split instruction, the destination register is a 64-bit
        # abstract register. Insert extract/zero-extend to extend the result
        # of the node to a 64-bit register.
        if not isinstance(orig_ty, BitVec):
            if not isinstance(orig_ty, Bool):
                logger.error(
                    f"What's this type? {orig_ty}, {sem.node_data(nidx, key='data')}"
                )
                assert False

            # Ok, the result of the tree is Bool, convert to bitvec and expand
            # to 64 bits.
            cvt = new_sem.add_cvt_bool_to_bv(new_sem.root)
            zext = new_sem.add_zext(cvt, 64)

            new_sem.add_node_data(cvt, "no_split", True)
            new_sem.add_node_data(zext, "no_split", True)

            # new_sem_r: Fix residual, convert from bv -> bool
            cvtr = new_sem_r.add_cvt_bv_to_bool(selected)
            for parent in parents:
                edge_idx = sem.edge_data(parent, nidx, key="idx")
                new_sem_r.add_edge(parent, cvtr, idx=edge_idx)
            new_sem_r.add_node_data(cvtr, "no_split", True)
        elif orig_ty.sz < 64:
            zext = new_sem.add_zext(new_sem.root, 64)
            new_sem.add_node_data(zext, "no_split", True)

            # new_sem_r: Add an extract
            extr = new_sem_r.add_extract(selected, 0, orig_ty.sz - 1)
            for parent in parents:
                edge_idx = sem.edge_data(parent, nidx, key="idx")
                new_sem_r.add_edge(parent, extr, idx=edge_idx)
                logger.info(f"Edge: {parent} {extr} {edge_idx}")
            new_sem_r.add_node_data(extr, "no_split", True)
        elif orig_ty.sz > 64:
            # Extract the lower 64 bits
            extr = new_sem.add_extract(new_sem.root, 0, 63)
            new_sem.add_node_data(extr, "no_split", True)

            # new_sem_r: zero-extend the selected register
            zextn = new_sem_r.add_zext(selected, orig_ty.sz)
            for parent in parents:
                edge_idx = sem.edge_data(parent, nidx, key="idx")
                new_sem_r.add_edge(parent, zextn, idx=edge_idx)
                logger.info(f"Edge: {parent} {zextn} {edge_idx}")

            new_sem_r.add_node_data(zextn, "no_split", True)
        else:
            for parent in parents:
                edge_idx = sem.edge_data(parent, nidx, key="idx")
                new_sem_r.add_edge(parent, selected, idx=edge_idx)

        return new_sem, new_sem_r

    def _used_regs_in_sem(self):
        wl = [self.root]
        used = []
        while wl:
            current = wl.pop(0)
            nd = self.node_data(current, "data")
            if isinstance(nd, str) and nd.startswith("R") and nd not in used:
                used.append(nd)
            wl.extend(self.successors(current))
        return used

    @property
    def root(self):
        roots = []
        for node in self.sem.nodes():
            if not list(self.predecessors(node)):
                roots.append(node)

        if len(roots) != 1:
            logger.error(f"Multiple/No root found: {len(roots)} {self.name}: {self.lhs}")
            self.semantics_to_svg()
            assert False

        return roots[0]

    # Conversion to external representation/solver stuff
    def to_claripy(self):
        # Ensure type information is present
        # TODO: This is not optimized.
        Typing.infer(self)
        return self.to_claripy_subtree(self.root)

    def to_claripy_subtree(self, root):
        current = self.node_data(root, "data")
        # If its just an integer constant, return it.
        if isinstance(current, int):
            return current

        succs_vars = []
        succs = list(self.successors_ordered(root))
        for succ in succs:
            stree = self.to_claripy_subtree(succ)
            succs_vars.append((stree, self.edge_data(root, succ, "idx")))

        succs_vars = [x[0] for x in sorted(succs_vars, key=lambda x: x[1])]
        if current == "extractMInt":
            sz = self.node_data(succs[0], Typing.KEY).sz
            succs_vars[1] = sz - succs_vars[1] - 1
            succs_vars[2] = sz - succs_vars[2]

        cp_func = maps.to_claripy(current)

        if not cp_func:
            assert not succs_vars, f"{current} is not a terminal!"
            sz = self.node_data(root, Typing.KEY).sz
            # Other arguments to claripy.BVS
            succs_vars = [current, sz, None, None, None, False, True]
            cp_func = claripy.BVS

        return cp_func(*succs_vars)

    # Pretty-print/display methods
    def semantics_to_dot(self, fname=None):
        from networkx.drawing.nx_pydot import write_dot

        if not fname:
            fname = tf.NamedTemporaryFile(suffix=".dot", delete=False).name
        write_dot(self.sem, fname)

    def semantics_to_svg(self, fname=None):
        from matplotlib import pyplot as plt
        from networkx.drawing.nx_agraph import graphviz_layout

        if not fname:
            fname = tf.NamedTemporaryFile(suffix=".svg", delete=False).name

        plt.figure(3, figsize=(12, 9))
        pos = graphviz_layout(self.sem, prog="dot", root=0, args="-Gsize=12,9")
        nx.draw(self.sem, pos, node_color="w", node_shape="s")
        node_labels = nx.get_node_attributes(self.sem, "data")
        for k, v in node_labels.items():
            node_labels[k] = f"%{k}: {v}"
        nx.draw_networkx_labels(self.sem, pos, labels=node_labels, font_size=10)
        plt.savefig(fname)
        plt.show()
        plt.close()

    def pretty_print_semantics(self):
        for node in nx.dfs_preorder_nodes(self.sem):
            print(node, self.sem.nodes[node])

    def pretty_print_prefix(self):
        roots = []
        # Find root
        for n in self.sem.nodes():
            if len(list(self.predecessors(n))) == 0:
                roots.append(n)

        for node in roots:
            res = self.pretty_print_recursive(node, 0)

        print("Prefix:", res)
        return res

    def pretty_print_recursive(self, nidx, depth):
        results = []
        succ_count = len(list(self.successors(nidx)))
        for idx, node in enumerate(self.successors(nidx)):
            results.append(self.pretty_print_recursive(node, depth + 1))
            if idx < succ_count - 1:
                results.append(",")

        ndata = self.node_data(nidx)["data"]
        if succ_count > 0:
            results.insert(0, str(ndata))
            results.insert(0, " (")
            results.append(")")
        else:
            results.insert(0, " " + str(ndata))

        rr = "".join(results)

        if depth > 0:
            rr = "\n" + "    " * depth + rr

        return rr

    def to_pseudo_asm(self, nidx):
        pass

    def _get_next_idx(self):
        self.nidx += 1
        return self.nidx

    def __deepcopy__(self, memo):
        res = Semantics(self.lhs)
        res.mapping = copy.deepcopy(self.mapping, memo)
        res.nidx = self.nidx
        res.sem = copy.deepcopy(self.sem, memo)
        res._instsem = self._instsem
        return res
