import uuid
import os
from collections import defaultdict
import re
import itertools

import yaml
from loguru import logger

from rosette.k_to_rkt import GenRKTInst, GenRKTMeta
import rosette.templates
from semantics.containers import ConstOperand
import synthesis.pseudo
from synthesis.task_result import SynthesisTaskResult


class FlagVerifier:
    def __init__(self, isa, pseudof):
        self.isa = isa
        self.id = str(uuid.uuid4().hex)[:4]
        self.verify = defaultdict(lambda: defaultdict(dict))

        self.pseudo = []
        if isinstance(pseudof, str):
            self.pseudo = synthesis.pseudo.load_pseudos_from_yaml(pseudof)
        else:
            self.pseudo = pseudof

    def verify_solutions(self, resultf):
        jresults = []
        with open(resultf) as fd:
            results = yaml.load(fd)

        for name, res in results.items():
            jresults.append(res.__dict__)

        results = jresults

        for res in results:
            synthr = SynthesisTaskResult.from_json(res)
            if not synthr.is_success():
                continue
            self.verify_solution(synthr)

        self.verify = dict(self.verify)
        save = dict()
        for name, res in self.verify.items():
            save[name] = dict(res)

        with open("verification-results-3.yaml", "w") as fd:
            yaml.dump(save, fd)

    def verify_solution(self, result, factors=[]):
        # Get the program
        if not result.is_success():
            return

        inst_name = result.spec
        spec_inst = None

        # First, find the instruction this is a result for
        if inst_name.startswith("pseudo"):
            for inst in factors:
                if inst.name == inst_name:
                    spec_inst = inst
                    break
        else:
            for inst in self.isa:
                if inst.name == inst_name:
                    spec_inst = inst
                    break

        if spec_inst is None:
            logger.error(f"Cannot find {inst_name} in any store!")
            return True

        flags = [
            flag
            for flag, fsem in spec_inst.iter_output_flags()
            if len(fsem.sem.nodes()) >= 2
        ]

        # Early return if there are no flags to synthesize
        if not flags:
            return True

        id = result.name.rsplit("-", 1)[1][:-4]

        logger.info(f"[{inst_name}] Checking flags for solution:\n{result.program}")

        prog = result.program.split("\n")
        opcodes = []
        proglist = []

        # Get opcodes
        for line in prog[1:]:
            opcode, tokens = line.split(" ", 1)

            if opcode == "movq":
                if tokens.startswith("0x"):
                    opcode = "MOVQ-IMM-R64"
                    tokens = re.sub(r"0x([0-9a-f]+)", r"(bv #x\1 64)", tokens)
                    operands = re.sub(r"(R([0-9]))", r"\2", tokens)
                    operands = re.sub(r",", "", operands)
                else:
                    opcode = "PMOVQ-R64-R64"
                    operands = tokens.strip()
                    operands = re.sub(r"(R([0-9]))", r"\2", operands)
                    operands = re.sub(r",", "", operands)
            elif opcode == "pnot":
                opcode = "notq-r64"
                operands = tokens.strip()
                operands = re.sub(r"(R([0-9]))", r"\2", operands)
                operands = re.sub(r",", "", operands)
            elif opcode == "psplit":
                opcode = "PSPLIT-R64-R64"
                operands = tokens.strip()
                operands = re.sub(r"(R([0-9]))", r"\2", operands)
                operands = re.sub(r",", "", operands)
            elif opcode in [
                "pmov-flag-r64",
                "pmov-r64-flag",
                "pset-flag",
                "preset-flag",
            ]:
                tokens = re.sub(r"F(\(.*\))", r"\1", tokens)
                operands = tokens.strip()
                operands = re.sub(r"(R([0-9]))", r"\2", operands)
                operands = re.sub(r",", "", operands)
            elif opcode in [
                "pxor-r64-r64",
                "pand-r64-r64",
                "por-r64-r64",
                "pnop", "pcmov-r64-r64-r64",
                "pconcat-r32-r32",
            ]:
                if opcode == "pxor-r64-r64":
                    opcode = "xorq-r64-r64"
                elif opcode == "por-r64-r64":
                    opcode = "orq-r64-r64"
                elif opcode == "pand-r64-r64":
                    opcode = "andq-r64-r64"
                operands = tokens.strip()
                operands = re.sub(r"(R([0-9]))", r"\2", operands)
                operands = re.sub(r",", "", operands)
            else:
                tokens, operands = tokens.rsplit("]", 1)
                operands = operands.strip()
                operands = re.sub(r"(R([0-9]))", r"\2", operands)
                operands = re.sub(r",", "", operands)

            operands = re.sub(r"\$1", "", operands)

            if opcode.startswith("pseudo-"):
                opcode = f"pseudo-{'-'.join(opcode.split('-')[1:]).upper()}"
            else:
                opcode = opcode.upper()

            proglist.append((opcode, operands))

            # opcode = line.split(' ')[0]
            if opcode not in opcodes:
                opcodes.append(opcode.lower())

        # Get the list of instructions part of the "solution"
        insts = [spec_inst]
        logger.debug(f"======== FACTORS: {[x.name for x in factors]}")
        for inst in itertools.chain(self.isa, factors):
            if inst.name.lower() not in opcodes:
                continue
            insts.append(inst)

        # Emit the corresponding rkt files and machine-meta for instructions that are a
        # part of the solution
        for inst_ in insts:
            _ = self.generate_inst_rkt(inst_)

        inst_dir = os.path.join("rosette", "inst_sems")
        _ = GenRKTMeta.RKT(inst_dir, insts, self.pseudo, suffix=self.id)

        # Emit verification file, with verify called as a function for the different flags
        # Parse output and collect results

        synth_task = self.generate_harness(spec_inst, insts, proglist, flags)
        result = self.do_task(synth_task)
        if result:
            self.verify[inst_name][id] = True
            logger.success(f"Success {synth_task} {flags}")
            return True
        else:
            self.verify[inst_name][id] = False
            logger.info(f"Failed {synth_task} {flags}")
            return False

    def generate_inst_rkt(self, instsem):
        logger.info(f"Generating RKT: {instsem.name}")
        inst_dir = os.path.join("rosette", "inst_sems")

        try:
            _ = GenRKTInst.RKT(inst_dir, instsem, flags=True)
        except NotImplementedError:
            return False

        return True

    def generate_harness(self, spec, insts, proglist, flags):
        comps = []
        spec_name = GenRKTInst.insn_to_struct_name(spec)

        prog_str = "\n".join([f"({x[0]} {x[1]})" for x in proglist])

        for inst in insts:
            comps.append(f'"inst_sems/{GenRKTInst.insn_to_struct_name(inst)}.rkt"')

        comps.append('"pseudo.rkt"')
        imports = [
            rosette.templates.IMPORTS,
            '(require "synthesis_core.rkt" "machine.rkt"',
            f'"inst_sems/machine-meta-{self.id}.rkt"',
            f'"inst_sems/{spec_name}.rkt"',
            "\n".join(comps),
            ")",
        ]

        prog = ["(list", prog_str, ")"]

        spec_ops = " ".join(
            [
                f"{i}"
                for i, op in enumerate(spec.operands)
                if not isinstance(op, ConstOperand) and not op.name.startswith("%")
            ]
        )

        asserts = [
            "(lambda (S S*)",
            f"(interpret-x86insn S ({spec_name} {spec_ops}))",
        ]

        for flag in flags:
            asserts.append(f"(assert (bveq (state-{flag} S) (state-{flag} S*)))")

        asserts.append(")")

        buf = [
            "\n".join(imports),
            "(verify-solution",
            "\n".join(prog),
            "\n".join(asserts),
            "interpret-x86insn",
            ")",
        ]

        bufstr = "\n".join(buf)
        path = os.path.join("rosette", f"synth-verify-{spec.name}-{self.id}.rkt")
        with open(path, "w") as fd:
            fd.write(bufstr)

        logger.info(f"Generated verification harness in {path}")
        return path

    def do_task(self, task):
        import subprocess as sp
        import shlex

        make = f"raco make {task}"
        ex = shlex.split(f"racket {task}")

        logger.info(f"Starting verify {task}")

        try:
            sp.check_call(make, shell=True)
        except sp.CalledProcessError as err:
            logger.warning(f"make failed: {task} {err}")
            return False

        try:
            proc = sp.Popen(ex, stdout=sp.PIPE, stderr=sp.PIPE, preexec_fn=os.setpgrp)
            outs, errs = proc.communicate(timeout=60)
        except sp.TimeoutExpired:
            logger.warning(f"Timeout: {task}")
            return False
        except sp.CalledProcessError as err:
            logger.warning(f"Exec failed ({task}): {err}")
            return False

        outs = outs.decode("utf-8").strip()
        errs = errs.decode("utf-8").strip()

        if outs != "#t":
            return False
        return True


if __name__ == "__main__":
    import argparse
    from synthesis.synthesis import SynthesisInstance

    argp = argparse.ArgumentParser()

    argp.add_argument(
        "--isa", nargs="+", type=str, help="Instruction part of current ISA"
    )

    argp.add_argument(
        "--pseudo-inst",
        default=None,
        type=str,
        help="Load pseudo instructions to use in synthesis from yaml file",
    )

    argp.add_argument("--result-file", type=str, help="Read results from result file")

    args = argp.parse_args()

    si = SynthesisInstance(args.isa, [], None, "knn")

    ff = FlagVerifier(si.sems, args.pseudo_inst)

    if args.result_file:
        ff.verify_solutions(args.result_file)
