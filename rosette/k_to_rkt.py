# This module implements the necessary classes and functions to interpret a
# parsed K-AST and convert to a <inst>.rkt file that:
# 1. Defines a struct for the corresponding <inst>,
# 2. Defines an interpretter for <inst> over bitvector logic

import os

from loguru import logger

from . import templates
import semantics.mappings as semmap
from semantics.containers import ConstOperand


class GenRKTInst:
    flags = ["CF", "PF", "AF", "ZF", "SF", "DF", "OF"]
    x86_rkt_regmap = {
        "RAX": 4,
        "RCX": 5,
        "RDX": 6,
    }

    def __init__(self, dir, inst, flags=False):
        self.dir = dir
        self.inst = inst
        self.rkt = []
        self.flags = flags

    @staticmethod
    def RKT(dir, inst, flags=False):
        grkt = GenRKTInst(dir, inst, flags)
        grkt._gen_imports()
        grkt._gen_exports()
        grkt._gen_struct_def()
        grkt._gen_print_function()
        grkt._gen_semantics()

        # grkt.debug_print()
        grkt.dump()

        return grkt

    def debug_print(self):
        print("\n".join(self.rkt))

    def dump(self):
        path = os.path.join(self.dir, GenRKTInst.insn_to_filename(self.inst))
        try:
            _ = open(path)
            logger.info(f"Using {path} from cache")
        except IOError:
            with open(path, "w") as fd:
                fd.write("\n".join(self.rkt))
            logger.info(f"Wrote: {path}")

    @staticmethod
    def insn_to_filename(insn):
        return f"{insn.name}.rkt"

    @staticmethod
    def insn_to_struct_name(insn):
        return insn.name

    @staticmethod
    def insn_to_struct_fields(insn):
        return " ".join(
            [
                f"{op.name.lower()}"
                for i, op in enumerate(insn.operands)
                if not isinstance(op, ConstOperand) and not op.name.startswith("%")
            ]
        )

    def _gen_struct_def(self):
        name = GenRKTInst.insn_to_struct_name(self.inst)
        operands = GenRKTInst.insn_to_struct_fields(self.inst)

        self.rkt.append(f"(struct {name} ({operands}) #:transparent)")

    def normalize_opname(self, opname):
        op = opname.lower()
        if op.startswith("r"):
            return op
        elif op.endswith("f"):
            return opname
        elif op.startswith("%"):
            return op[1:]

    def _gen_semantics(self):
        # NOTE: This has to be in lockstep with the machine definition in
        # rosette/machine.rkt

        argstr = []
        for op in self.inst.operands:
            if isinstance(op, ConstOperand):
                continue
            if op.name.startswith("%"):
                continue
            argstr.append(op.name.lower())
        argstr = " ".join(argstr)
        func_decl = f"(define (interpret-{self.inst.name} S {argstr})"

        # Create a let binding to read all operands; This prevents the
        # computation's state update messing up future computations
        let_bindings = ["(let ("]
        for op in self.inst.operands:
            if isinstance(op, ConstOperand) or op.name.startswith("%"):
                continue
            opname = self.normalize_opname(op.name)
            bind = f"[local-{opname} (state-Rn-ref S {opname})]"
            let_bindings.append(bind)

        # Do the same for flags
        for op in GenRKTInst.flags:
            bind = f"[local-{op} (state-{op} S)]"
            let_bindings.append(bind)

        # Conservatively read hardware registers
        for op in ["rax", "rcx", "rdx"]:
            bind = f"[local-{op} (read-hw-{op} S)]"
            let_bindings.append(bind)

        let_bindings.append(")")

        # Iterate over all the output registers
        func_body = []

        # Always do registers first, before the flags
        for reg, sem in self.inst.sems.items():
            if self.inst.is_flag_reg(reg):
                continue

            # Check if the output is a hardware register or an abstract
            # register
            if reg in GenRKTInst.x86_rkt_regmap:
                func_preamble = f"(write-hw-{reg.lower()}! S (interpret-k"
            else:
                func_preamble = f"(state-Rn-set! S {reg.lower()} (interpret-k"

            func_body.append(func_preamble)
            root = sem.root
            func_sem = self._gen_semantics_recursive(sem, root)
            func_body.append(func_sem)
            func_body.append("))")

        # Now, do the flags
        for reg, sem in self.inst.sems.items():
            if not self.inst.is_flag_reg(reg):
                continue

            if not self.flags:
                continue

            func_preamble = (
                f"(state-F-set! S {GenRKTInst.flags.index(reg)} (interpret-k"
            )
            func_body.append(func_preamble)
            root = sem.root
            func_sem = self._gen_semantics_recursive(sem, root)
            func_body.append(func_sem)
            func_body.append("))")

        if not func_body:
            func_body = ["(void)"]

        if func_body:
            func_body = [*let_bindings, "(begin", *func_body, "))"]

        func_end = ")"

        self.rkt.extend([func_decl, *func_body, func_end])

    def _gen_semantics_recursive(self, sem, idx):
        subtrees = []
        for succ in sem.successors_ordered(idx):
            subtrees.append(self._gen_semantics_recursive(sem, succ))

        nd = sem.node_data(idx, "data")
        current = semmap.to_rosette(nd)

        if current is None:
            # Check if it is an integer constant
            try:
                current = f"{int(nd)}"
            except ValueError:
                # Treat it as a register/flag and lookup the value of it in
                # the state variable
                if nd.startswith("undefMInt"):
                    current = f"(undefine 1)"
                elif nd.startswith("undefBool"):
                    current = "(undef-bool)"
                elif nd == "false":
                    current = "#f"
                elif nd == "true":
                    current = "#t"
                elif nd.startswith("R"):
                    opname = self.normalize_opname(nd)
                    current = f"local-{opname}"
                elif nd.endswith("F"):
                    opname = self.normalize_opname(nd)
                    current = f"local-{opname}"
                elif nd.startswith("%"):
                    # This is a fixed hardware register
                    # These reads/writes need to be defined in machine.rkt
                    opname = self.normalize_opname(nd)
                    current = f"local-{opname}"
                else:
                    logger.warning(f"[x] What's this?! {nd}")
                    raise NotImplementedError

        if subtrees:
            return f"({current} {' '.join(subtrees)})"
        else:
            return current

    def _gen_imports(self):
        self.rkt.append(templates.IMPORTS)
        self.rkt.append('(require "../machine.rkt" "../k.rkt")')

    def _gen_exports(self):
        export = "(provide (all-defined-out))"
        self.rkt.append(export)

    def _gen_print_function(self):
        opstr = GenRKTInst.insn_to_struct_fields(self.inst)

        optyps = ", ".join([f"{op.name}: {op.sz}" for op in self.inst.operands])

        fmtstr = []
        for op in self.inst.operands:
            # Hardware register, skip/ignore in priting
            if op.name.startswith("%"):
                continue
            elif isinstance(op, ConstOperand):
                fmtstr.append(f"{op}")
            else:
                fmtstr.append("R~s")

        if fmtstr:
            fmtstr[-1] = "R~s~n"
        fmtstr = ", ".join(fmtstr)

        self.rkt.extend(
            [
                f"(define (print-{self.inst.name} {opstr})",
                f'  (printf "{self.inst.name.lower()} [{optyps}] {fmtstr}" {opstr}))',
            ]
        )


class GenRKTMeta:
    def __init__(self, out, insns, pseudo=[], suffix="1"):
        self.out = out
        self.insns = insns
        self.rkt = []
        self.pseudos = pseudo
        self.suffix = suffix

    @classmethod
    def RKT(cls, out, insns, pseudo=[], suffix="1"):
        grkt = GenRKTMeta(out, insns, pseudo, suffix)

        grkt._gen_imports()
        grkt._gen_exports()
        grkt._gen_printer()
        grkt._gen_interpretter()
        grkt.dump()

        # grkt.debug_print()

        return grkt

    def dump(self):
        path = os.path.join(self.out, f"machine-meta-{self.suffix}.rkt")
        logger.info(f"Wrote machine-meta to: {path}")
        with open(path, "w") as fd:
            fd.write("\n".join(self.rkt))

    def debug_print(self):
        print("\n".join(self.rkt))

    def _gen_imports(self):
        self.rkt.append(templates.IMPORTS)
        if not self.insns:
            return

        self.rkt.append("(require ")

        for insn in self.insns:
            if not insn.is_pseudo():
                path = GenRKTInst.insn_to_filename(insn)
                self.rkt.append(f'"{path}"')
            else:
                path = f'"../{insn.defined_in}"'
                self.rkt.append(path)

        # Imports for pseudo instructions
        pi = []
        for pseudo in self.pseudos:
            # XXX: This is hacky way to compute path and brittle if file
            # organization changes in the future
            fname = f'"../{pseudo.defined_in}"'
            if fname not in pi:
                pi.append(fname)

        self.rkt.extend(pi)

        self.rkt.append(")")

    def _gen_exports(self):
        export = "(provide (all-defined-out))"
        self.rkt.append(export)

    def _gen_printer(self):
        self.rkt.extend(["(define (print-x86insn insn)", "(match insn"])

        for insn in self.insns:
            struct = GenRKTInst.insn_to_struct_name(insn)
            ops = GenRKTInst.insn_to_struct_fields(insn)
            self.rkt.append(f"[({struct} {ops}) (print-{insn.name} {ops})]")

        # Go through all the pseudo cases
        for pseudo in self.pseudos:
            struct = pseudo.struct_name
            ops = " ".join([x.name for x in pseudo.operands])
            self.rkt.append(f"[({struct} {ops}) ({pseudo.print_fn} {ops})]")

        self.rkt.append("))")

    def _gen_interpretter(self):
        self.rkt.extend(["(define (interpret-x86insn S insn)", "(match insn"])

        for insn in self.insns:
            struct = GenRKTInst.insn_to_struct_name(insn)
            ops = GenRKTInst.insn_to_struct_fields(insn)
            self.rkt.append(f"[({struct} {ops}) (interpret-{insn.name} S {ops})]")

        # Go through all the pseudo cases
        for pseudo in self.pseudos:
            struct = pseudo.struct_name
            ops = " ".join([x.name for x in pseudo.operands])
            self.rkt.append(f"[({struct} {ops}) ({pseudo.interpret_fn} S {ops})]")

        # Have a catch-all case
        self.rkt.append(f"[v v]")

        self.rkt.append("))")


if __name__ == "__main__":
    import argparse
    from k_parser.parse_k import SimplParser

    argp = argparse.ArgumentParser()
    argp.add_argument(
        "insts", nargs="+", type=str, help="Instruction to convert to rkt"
    )

    argp.add_argument("-o", type=str, help="Output directory")
    argp.add_argument(
        "--flags", action="store_true", help="Include flag definitions in semantics"
    )

    args = argp.parse_args()

    parsed = []

    for inst in args.insts:
        sp = SimplParser(inst)
        try:
            sp.do_parse()
            sp.simplify_semantics()
        except AssertionError as err:
            logger.warn(f"[x] Failed on: {inst}, {err}")
            continue

        if "R2" in sp.instsem.sems:
            r2 = sp.instsem.sems["R2"]
            _ = GenRKTInst.RKT(args.o, sp.instsem, flags=args.flags)
        else:
            logger.warn(f"[x] No R2: {args.inst}")
            continue

        parsed.append(sp.instsem)

    _ = GenRKTMeta.RKT(args.o, parsed)
