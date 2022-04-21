import argparse
import re
import glob
import tqdm
from loguru import logger

from k_parser.grammar import (
    # kgrammar,
    cf_opcodes,
    bool_opcodes,
    functions,
    floating_point_functions,
    all_functions,
)
import semantics.containers as cont
from semantics.typing import Typing


class SimplParser:
    def __init__(self, kfile):
        self.kfile = kfile
        self.sems = dict()
        self.nidx = 0
        self.last_pop = None
        self.instsem = None

    def do_parse(self):
        with open(self.kfile) as fd:
            sem = fd.read()

        cursem = None
        modline = re.compile(r"module (?P<name>[-\w]+)")
        lines = iter(sem.split("\n"))

        for line in lines:
            # Skip till modline start
            m = re.search(modline, line)
            if not m:
                continue

            instruction = m.groups("name")[0]
            break

        assert instruction is not None, "Module start not found!"

        for line in lines:
            line = line.strip()
            if line.startswith("execinstr"):
                if not cursem:
                    opcode, operands = self.parse_exec_instr(line)
                    cursem = cont.InstructionSemantics(instruction, opcode, operands)
                else:
                    continue
                    # cursem.add_alias(opcode, operands)

            if line.startswith("<regstate>"):
                break

        # Deal with regstate/actual semantics for updating register values
        seen_keys = set()
        for line in lines:
            if "|->" in line:
                lhs, rhs = line.split("|->")
                lhs = lhs.strip()
                if '"' in lhs:
                    lhs = lhs[1:-1]

                if lhs in seen_keys:
                    logger.warning("Multiple definitions for {lhs}; Skipping!")
                    continue

                seen_keys.add(lhs)
                sem = self.parse_semantics(lhs, rhs)
                cursem.define_semantics_for(lhs, sem)

        # We're done with phase 1 of parsing, save to instsem
        self.instsem = cursem
        self.simplify_semantics()
        self.populate_metadata()

        # print(cursem)
        # cursem.semantics_to_dot()
        # cursem.pretty_print_semantics()
        # cursem.pretty_print_prefix()

        return self.instsem

    def parse_bool_opcode(self, key, tok, wl):
        sem = self.sems[key]
        # Handle the unary case
        if tok == "notBool":
            # The next expression that follows this token is the operand to
            # notBool.
            this_idx = self.new_node(sem, tok)
            self.add_as_child(sem, wl, this_idx)
            self.step_in(wl, tok, this_idx, sem)
            return

        # Now, we're in the case of binary operations.
        # This needs some graph manipulation: We need to take the root of the
        # previous expression and add as a child to the bool operation.
        last = self.last_pop
        this_idx = self.new_node(sem, tok)

        # Redirect all incoming edges to last into this_idx
        sem.redirect_incoming(last, this_idx)
        self.add_operand(sem, this_idx, last, 0)

        self.step_in(wl, tok, this_idx, sem)

    def parse_cf_opcode(self, key, tok, wl):
        sem = self.sems[key]
        if tok == "#ifMInt":
            this_idx = self.new_node(sem, tok)
            self.add_as_child(sem, wl, this_idx)
            self.step_in(wl, tok, this_idx, sem)
        elif tok == "#fi":
            self.step_out_till_if(wl, sem)
            self.step_out(wl, 1, sem)
        elif tok == "#else":
            self.step_out_till_if(wl, sem)
            this_idx = self.new_node(sem, tok)
            self.add_as_child(sem, wl, this_idx)
            self.step_in(wl, tok, this_idx, sem)
        elif tok == "#then":
            self.step_out_till_if(wl, sem)
            this_idx = self.new_node(sem, tok)
            self.add_as_child(sem, wl, this_idx)
            self.step_in(wl, tok, this_idx, sem)

    def parse_function(self, key, tok, wl):
        sem = self.sems[key]
        this_idx = self.new_node(sem, tok)
        self.add_as_child(sem, wl, this_idx)
        self.step_in(wl, tok, this_idx, sem)

    def parse_semantics(self, key, rhs):
        sexp = re.compile(r"([()])|([\w]*[%=A-Za-z0-9#\"]+)|([,])")
        tokens = re.split(sexp, rhs)

        # Reset class variables to start new round of parsing
        sem = cont.Semantics(key)
        self.sems[key] = sem
        self.last_pop = None

        self.nidx = 0
        sem.mapping["("] = {"data": "("}
        sem.mapping[")"] = {"data": ")"}
        worklist = []

        for tok in tokens:
            if not tok or tok == " ":
                continue

            tok = tok.strip()
            if '"' in tok:
                tok = tok[1:-1]

            # print(f"Current is: {tok}")
            # Check if the token is an integer
            value = None
            try:
                value = int(tok)
            except ValueError:
                pass

            if value is not None:
                this_idx = self.new_node(sem, value)
                self.add_as_child(sem, worklist, this_idx)
            elif tok in bool_opcodes:
                self.parse_bool_opcode(key, tok, worklist)
            elif tok in cf_opcodes:
                self.parse_cf_opcode(key, tok, worklist)
            elif tok in all_functions:
                self.parse_function(key, tok, worklist)
            elif tok == "(":
                self.step_in(worklist, "(", "(", sem)
            elif tok == ")":
                # step-out till a matching brace is found
                x = self.step_out_one_brace(worklist, sem)
                assert x == "(", f"Closing paren matched {x}"
                # Check if top of wl is a function.
                if worklist and sem.lookup_mapping(worklist[-1]) in functions:
                    self.step_out(worklist, 1, sem)
            elif tok == ",":
                # This is also a no-op, as commas pretty much need to
                # increment the edge index.
                pass
            else:
                # print(f"Missed: {tok}, treating as a terminal")
                this_idx = self.new_node(sem, tok)
                self.add_as_child(sem, worklist, this_idx)

        assert len(worklist) == 0, f"Reduction failed? {len(worklist)}"
        return sem

    def new_node(self, sem, tok):
        idx = sem.add_node(data=tok)
        return idx

    def add_as_child(self, sem, wl, nn):
        if not wl:
            return
        cidx = len(wl) - 1
        while cidx >= 0 and wl[cidx] == "(":
            cidx -= 1
        if cidx >= 0:
            self.add_operand(sem, wl[cidx], nn)

    def add_operand(self, sem, source, target, idx=None):
        if not idx:
            idx = len(list(sem.successors(source)))
        sem.add_edge(source, target, idx=idx)

    def step_out_till_if(self, wl, sem):
        top = sem.lookup_mapping(wl[-1])
        while not top.startswith("#if"):
            self.step_out(wl, 1, sem)
            top = sem.lookup_mapping(wl[-1])

    def step_out_one_brace(self, wl, sem):
        # print(sem.lookup_mapping(wl[-1]))
        top = sem.lookup_mapping(wl[-1])
        while not top.startswith("("):
            self.step_out(wl, 1, sem)
            top = sem.lookup_mapping(wl[-1])
        return self.step_out(wl, 1, sem)

    def step_out(self, wl, count, sem):
        for i in range(count):
            n = wl.pop(-1)
            if n != "(":
                self.last_pop = n
            # print(f"\tStepping out: {n} {sem.lookup_mapping(n)}")
            if wl:
                pass
                # print(f"\t\t-TOP: {wl[-1]} {sem.lookup_mapping(wl[-1])}")
        return n

    def step_in(self, wl, tok, nidx, sem):
        wl.append(nidx)
        # print(f"\tStepping in: {nidx} {sem.lookup_mapping(nidx)}")

    def parse_exec_instr(self, line):
        # Check if there is a recursive redefinition.
        # TODO: Assuption, the RHS will reappear as LHS sometime in the future
        # and we catch that definition there.
        if "=>" in line:
            lhs, rhs = [x.strip() for x in line.split("=>")]

        _, line = lhs.split("(")
        elements = re.split(r"[, )]+", line)
        # print(elements)

        opcode = elements[0]
        operands = []
        for elem in elements[1:]:
            elem = elem.strip()
            if not elem:
                continue
            if "$" in elem:
                # Handle constants
                value = int(elem[1:], 16)
                op = cont.ConstOperand(value)
                operands.append(op)
            elif ":" in elem:
                reg, ty = elem.split(":")
                try:
                    sz = int(ty[1:])
                    ty = ty[0]
                except ValueError:
                    if ty == "Ymm":
                        sz = 256
                    elif ty == "Xmm":
                        sz = 128
                    elif ty == "Rh":
                        sz = 8
                    elif ty == "Rl":
                        sz = 8
                    else:
                        assert False, f"Unknown operand type, found: {ty}"
                op = cont.Operand(reg, sz, ty)
                operands.append(op)
            elif elem.startswith("%"):
                # TODO: Get/implement a register file for lookup
                reg = elem
                if elem in ["%cl", "%al"]:
                    ty = "Gpr"
                    sz = 8
                elif elem in ["%cx", "%ax"]:
                    ty = "Gpr"
                    sz = 16
                elif elem in ["%ecx", "%eax"]:
                    ty = "Gpr"
                    sz = 32
                elif elem in ["%rcx", "%rax"]:
                    ty = "Gpr"
                    sz = 64
                elif elem.startswith("%xmm"):
                    ty = "Xmm"
                    sz = 128
                else:
                    ty = "R"
                    # TODO: Find out operand size from register size
                    sz = "Unknown"
                op = cont.Operand(reg, sz, ty)
                operands.append(op)
            elif elem == ".Operands":
                # Noop
                pass
            else:
                assert False, f"Unknown operand type, found: {elem}"

        return opcode, operands

    def infer_types(self, normalized=False):
        """
        Calls the type inference pass over the instruction semantics
        """
        for lhs in self.instsem.iter_semantics():
            sem = self.instsem.sems[lhs]
            Typing.infer(sem, normalized=normalized)

    def simplify_semantics(self):
        """
        Iterate over semantics, and do minor simplifications
        This is the top-level function to perform simplifications on all lhs
        variables.
        """
        for lhs in list(self.instsem.iter_semantics()):
            self.simplify_semantics_for(lhs)

    def populate_metadata(self):
        # Check if the instruction has vector operands
        for op in self.instsem.operands:
            if isinstance(op, cont.ConstOperand):
                continue

            if op.sz > 64:
                self.instsem.set_metadata("vec", True)

        for lhs in self.instsem.iter_semantics():
            sem = self.instsem.sems[lhs]
            for node in sem.sem.nodes():
                nty = sem.node_data(node, "data")
                if nty in floating_point_functions:
                    self.instsem.set_metadata("float", True)

    def simplify_semantics_for(self, key):
        """
        Simplify semantics for a single `lhs` variable

        :key: str = lhs value whose correspoding semantics should be
        simplified.
        """
        sem = self.instsem.sems[key]

        # Check if lhs is of the form: convToReg
        if key.startswith("convToRegKeys"):
            newk = key[len("convToRegKeys") + 1 : -1]
            self.instsem.undef_semantics_for(key)
            self.instsem.define_semantics_for(newk, sem)

        # Now, walk over the AST and simplify some nodes
        # Convert it to a list to prevent mutation while traversing
        for node in list(sem.iter_postorder()):
            nty = sem.node_data(node, "data")
            # print(node, nty)
            if nty in ["getParentValue", "getFlag", "getRegisterValue"]:
                # If the node is a getParentValue, take its left operand,
                # which is a (abstract) register and replace this node by it.
                # * Same for "getFlag"
                # * Same for "getRegisterValue"
                abreg, rsmap = list(
                    sorted(
                        sem.successors(node),
                        key=lambda x: sem.edge_data(node, x, "idx"),
                    )
                )
                sem.redirect_incoming(node, abreg)
                sem.remove_edge(node, rsmap)
                sem.remove_edge(node, abreg)
                sem.remove_node(node)
                sem.remove_node(rsmap)

            if nty in ["#then", "#else"]:
                # Remove superflous then and else tags
                body = list(sem.successors(node))
                if len(body) != 1:
                    logger.error("#then or #else body has nodes != 1?!")
                    continue
                body = body[0]
                sem.redirect_incoming(node, body)
                sem.remove_edge(node, body)
                sem.remove_node(node)


def to_claripy(sem):
    r2 = sem.sems["R2"]
    cl = r2.to_claripy_subtree(0)
    print(cl)


def do_work(files):
    pbar = tqdm.tqdm(files)
    for kf in pbar:
        # pbar.set_description(f"current: {kf}")
        # pbar.refresh()
        print(kf)
        sp = SimplParser(kf)
        try:
            sp.do_parse()
            sp.simplify_semantics()
            sp.infer_types()
        except AssertionError as err:
            print("[x] Failed on:", kf, err)

        # to_claripy(sp.instsem)


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("input", type=str, help="Path to semantics file")
    argp.add_argument(
        "-d", action="store_true", help="Operate on a directory of K-files"
    )
    args = argp.parse_args()

    if args.d:
        inputs = glob.glob(f"{args.input}/*.k")
    else:
        inputs = [args.input]

    do_work(inputs)
