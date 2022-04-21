# Implements a synthesis task --- A worker that takes processes a queue of synthesis tasks

import os
import uuid
import subprocess as sp
from collections import defaultdict
import time
import shlex
import queue
import signal

from loguru import logger

from rosette.k_to_rkt import GenRKTInst, GenRKTMeta
import rosette.templates
from synthesis.task_result import SynthesisTaskResult


class SynthesisTask:
    def __init__(self):
        pass


class RosetteSynthesisTask(SynthesisTask):
    BASE_DIR = "rosette"

    def __init__(
        self,
        inst,
        components,
        max_prog_len,
        timeout,
        priority=10,
        pseudo=[],
        out="rosette",
        flag_task=False,
    ):
        self.timeout = timeout
        self.out = out
        self.id = str(uuid.uuid4().hex)[:4]
        self.flag_task = flag_task
        self.selector_name = None

        self.pseudo = pseudo

        self.spec = inst
        self.components = components

        self.max_prog_len = max_prog_len

        self.priority = priority
        self.tags = set()

        self.forced_comps = []
        self.blacklist_comps = []

    @property
    def name(self):
        if self.flag_task:
            return f"{self.spec.name}-flags-{self.max_prog_len}-{self.id}"
        return f"{self.spec.name}-{self.max_prog_len}-{self.id}"

    def add_tag(self, tag):
        self.tags.add(tag)

    def do_task(self):
        # Generate files for all the components
        # TODO: Cache this once things are stable
        for idx, component in enumerate(self.components):
            res = self.generate_inst_rkt(component)
            if not res:
                self.components[idx] = None

        # Only do this when the `spec` is a real instruction and not a pseudo instruction
        if "pseudo_inst" not in self.tags:
            self.generate_inst_rkt(self.spec)

        # Filter all the failed components
        self.components = [x for x in self.components if x is not None]

        synth_task = self._gen_harness(self.spec, self.components, self.max_prog_len)

        # Generate Meta files
        inst_dir = os.path.join(self.out, "inst_sems")
        _ = GenRKTMeta.RKT(
            inst_dir, [self.spec] + self.components, self.pseudo, suffix=self.id
        )

        # Spawn subprocess to invoke synthesis
        return self.execute_st(synth_task)

    def execute_st(self, synth_task):
        make = f"raco make {synth_task}"
        ex = shlex.split(f"racket {synth_task}")

        try:
            sp.check_call(make, shell=True)
        except sp.CalledProcessError as err:
            logger.warning(f"Make failed: {synth_task}")
            return SynthesisTaskResult(self, "error", synth_task, debug=err,
                                       is_flag_result=self.flag_task)

        status = None
        # Collect some stats
        synth_start = time.time()
        try:
            proc = sp.Popen(ex, stdout=sp.PIPE, stderr=sp.PIPE, preexec_fn=os.setpgrp)
            outs, errs = proc.communicate(timeout=self.timeout)
            status = "success"
        except sp.TimeoutExpired:
            logger.warning("Timeout!")
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            outs, errs = proc.communicate()
            status = "timeout"
        except sp.CalledProcessError as err:
            logger.warning(f"Exec failed: {err}")
            return SynthesisTaskResult(self, "error", synth_task, debug=err,
                                       is_flag_result=self.flag_task)
        except KeyboardInterrupt:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            raise KeyboardInterrupt

        synth_end = time.time()

        outs = outs.decode("utf-8").strip()
        errs = errs.decode("utf-8").strip()

        if status == "success":
            # Inspect outs
            for line in outs.split("\n"):
                line = line.strip()
                if not line:
                    status = "unsat"
                else:
                    break

        logger.info(f"{synth_task}; {status};\n{outs};\n{errs}")

        return SynthesisTaskResult(
            self,
            status,
            synth_task,
            data=outs,
            debug=errs,
            time=synth_end - synth_start,
            is_flag_result=self.flag_task
        )

    def generate_inst_rkt(self, instsem):
        logger.info(f"Generating RKT: {instsem.name}")
        inst_dir = os.path.join(self.out, "inst_sems")

        try:
            _ = GenRKTInst.RKT(inst_dir, instsem, flags=True)
        except NotImplementedError:
            return False

        return True

    def _gen_harness(self, spec, components, max_prog_len):
        inst = GenRKTInst.insn_to_struct_name(spec)
        instops = GenRKTInst.insn_to_struct_fields(spec)
        comps = []
        comps_req = []

        for comp in components:
            cinst = GenRKTInst.insn_to_struct_name(comp)
            cops = GenRKTInst.insn_to_struct_fields(comp)
            comps.append(f"({cinst} {cops})")

            comps_req.append(f'"inst_sems/{cinst}.rkt"')

        if "pseudo_inst" not in self.tags:
            comps_req.append(f'"inst_sems/{inst}.rkt"')
        else:
            comps_req.append(f'"{spec.defined_in}"')

        comps = "\n".join(comps)
        comps_req = "\n".join(comps_req)

        # Do the same thing for loaded pseudo instructions
        pimports = []
        pcomps = []
        for pseudo in self.pseudo:
            impstr = f'"{pseudo.defined_in}"'
            if impstr not in pimports:
                pimports.append(impstr)

            pcomps.append(
                f"({pseudo.struct_name} {' '.join([x.name for x in pseudo.operands])})"
            )

        pimports = "\n".join(pimports)
        pcomps = "\n".join(pcomps)

        imports = [
            rosette.templates.IMPORTS,
            '(require "synthesis_core.rkt" "machine.rkt"',
            f'"inst_sems/machine-meta-{self.id}.rkt"',
            comps_req,
            pimports,
            ")",
        ]

        # Generate spec and the corresponding asserts
        asserts = [
            f"(lambda (spec r1 r2 r3 r4 r5)",
            "(lambda (S S*)",
            "(define (synth-objective spec)",
            "(spec S r1 r2 r3 r4 r5))",
            "(begin",
            "(synth-objective spec)",
        ]

        # Check if the synthesis task is for flags or registers
        if self.flag_task:
            flags = []
            if isinstance(self.flag_task, list):
                flags = self.flag_task
            else:
                for flag, fsem in spec.iter_output_flags():
                    if len(fsem.sem.nodes()) < 2:
                        continue
                    flags.append(flag)
            for flag in flags:
                spec_flag_rd = f"(state-{flag} S)"
                synth_flag_rd = f"(state-{flag} S*)"
                asserts.append(f"(assert (bveq {spec_flag_rd} {synth_flag_rd}))")
        else:
            for reg, _ in spec.iter_output_regs():
                reg_rd = reg
                spec_reg_rd = f"(state-Rn-ref S {reg_rd.lower()})"
                synth_reg_rd = f"(state-Rn-ref S* {reg_rd.lower()})"

                if reg in GenRKTInst.x86_rkt_regmap:
                    spec_reg_rd = f"(read-hw-{reg_rd.lower()} S)"
                    synth_reg_rd = f"(read-hw-{reg_rd.lower()} S*)"

                asserts.append(f"(assert (bveq {spec_reg_rd} {synth_reg_rd}))")

        asserts.append(")))")

        asserts = "\n".join(asserts)

        rkt = [
            "\n".join(imports),
            f"(synth-insn-single",
            f"(lambda (S r1 r2 r3 r4 r5) (interpret-x86insn S ({inst} {instops})))",
            f"(list 0 1 2 4 5)",
            f"(lambda (r1 r2 r3 r4 r5 i)",
            f"(list {comps}\n{pcomps}))",
            f"{max_prog_len}",
            f"{asserts}",
            f"interpret-x86insn",
            f"print-x86insn",
            f")",
        ]

        path = os.path.join(self.out, f"synth-target-{inst}-{self.id}.rkt")
        with open(path, "w") as fd:
            fd.write("\n".join(rkt))

        logger.info(f"Generated harness in {path}")
        return path


def spawn_synthesis_worker(wq, result_queue):
    from synthesis.subgoals import MAX_TIMEOUT
    import synthesis.work_queue

    if isinstance(wq, synthesis.work_queue.ListWQ):
        work_queue = wq
    else:
        work_queue = synthesis.work_queue.BeanstalkWQ(wq)

    # work_queue: mp.Queue / BeanstalkWQ that contains objects of type `SynthesisTask`
    # A worker listens on the queue and pops an available synthesis task to process.
    # result_queue: mp.Queue / BeanstalkWQ. SynthesisTaskResult.
    logger.info("Worker ready to process synthesis task!")

    while True:
        try:
            task = work_queue.get(timeout=2 * MAX_TIMEOUT)
        except queue.Empty:
            logger.warning("Shutting down process as work_queue is empty!")
            break
        except KeyboardInterrupt:
            logger.warning("Stopping synthesis worker!")
            break

        logger.info(f"Starting synthesis task: {task.name}")

        try:
            result = task.do_task()
            work_queue.delete(task.job_id)
        except KeyboardInterrupt:
            logger.warning(f"Stopping synthesis task: {task.name}")
            break

        result_queue.put((result, task))


if __name__ == "__main__":
    pass
