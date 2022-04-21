# Implements the frontend/controller center for synthesis
#
# The idea is to keep this part of synthesis generic and have multiple
# ways/backends to emit tasks for different synthesis engines, e.g., Rosette

import argparse
import uuid
import yaml
from collections import defaultdict
import json


from loguru import logger
from pymongo import MongoClient

from k_parser.parse_k import SimplParser

import synthesis.pseudo
from synthesis.synthcomp import selector
import synthesis.work_queue
from synthesis.synth_task import RosetteSynthesisTask
from synthesis.transformations import Transformer
from synthesis.subgoals import WorkflowManager
from synthesis.factorization import FACTORIZERS
from synthesis.rewrites.rewrites import Rewriter


class SynthesisInstance:
    # Default MAX_TIMEOUT is 1d
    MAX_TIMEOUT = 2 * 24 * 60 * 60
    # Default maximum program length Maximum
    MAX_PROG_LEN = 5

    def __init__(
        self,
        isa,
        local_wq,
        pseudof,
        select_strat,
        redo_failures=None,
        parallel_tasks=1,
        try_factorization=False,
        factorizer="bottom_up",
        timeout=None,
        synthesize_pseudo=False,
        synthesize_flags=False,
        conn=None,
        long_run=False,
        iterative_rewrites=False,
    ):
        self.id = str(uuid.uuid4().hex)[:4]

        # Save logger to file
        logger.add(f"synth-results/synth-instance-{self.id}.log", enqueue=True)
        logger.success(f"Started Synthesis Instance: {self.id}")

        # Initialize all the class variables with default values
        self.sems = []
        self.semmap = dict()

        # Local work queue to bootstrap and scope the synthesis instance
        self.local_wq = []

        self.iterative_rewrites = iterative_rewrites

        # Shared work/result queue with the worker processes to process synthesis
        # tasks
        self.conn = None
        if conn is None and long_run:
            self.conn = {
                'beanstalkd': {'host': '127.0.0.1',
                               'port': 11300,
                               'wq': 'synthct'},
                'mongodb': {'host': '127.0.0.1',
                            'port': 27019,
                            'db': 'synthct'},
                'redis': {'host': '127.0.0.1',
                          'port': 6379}
            }
        elif conn:
            with open(conn) as fd:
                self.conn = yaml.safe_load(fd)

        # Initialize a connection to pymongo
        if self.conn:
            self.client = MongoClient(
                host=self.conn['mongodb']['host'],
                port=self.conn['mongodb']['port'])
            self.db = self.client[self.conn['mongodb']['db']]
            self.collection = self.db.create_collection(f"si-{self.id}", capped=True, size=20000000000)
        if self.conn:
            self.wq = synthesis.work_queue.BeanstalkWQ(self.conn)
        else:
            self.wq = synthesis.work_queue.ListWQ()

        self.result_q = synthesis.work_queue.ListWQ()

        self.synthesize_pseudo = synthesize_pseudo
        self.synthesize_flags = synthesize_flags

        self.pseudo = []
        self.selector = select_strat
        self.redo_failures = redo_failures
        self.skiplist = []
        self.result_queue = None
        self.parallel_tasks = parallel_tasks

        # Factorization related book keeping
        self.try_factorization = try_factorization
        self.factorizer = factorizer

        self.timeout = timeout or self.TIMEOUT
        self.max_prog_len = self.MAX_PROG_LEN
        self.max_comp_count = 32

        # Some book keeping and result tracking
        self.status = dict()

        # Bookkeeping on isa stats
        self.isa_stats = defaultdict(list)

        self.do_init(isa)

        if not synthesize_pseudo:
            for sem in self.sems:
                if local_wq:
                    if sem.name in local_wq:
                        self.local_wq.append(sem.name)
                else:
                    self.local_wq.append(sem.name)
            # Also load the pseudo instruction factors from local_wq
            for name in local_wq:
                if name.startswith("pseudo-"):
                    self.local_wq.append(name)

        # Process pseudo instruction if they're loaded
        if pseudof:
            self.pseudo = synthesis.pseudo.load_pseudos_from_yaml(pseudof)

        # Filter if redo-failures is set
        if self.redo_failures:
            self.filter_success(redo_failures)

        logger.info(f"Created SynthesisInstance-{self.id}")

    def do_init(self, isa):
        # =============================
        # Filtering instructions:
        # =============================
        # For the first phase, to simplify our life, we disallow some
        # instructions. We will relax these constraints to scale to full ISA
        # later on:
        # 1. No Floating Point
        # 2. No Vector instructions; our current machine model does not
        #    support it, although it would be trivial to implement it.
        # 3. Instructions that fail parsing for a variety of reasons.
        #    3.1. Depending on which instruction and the reason, this is the
        #         first place to start fixing instructions.
        # 4. Exclude synthesis of all type of movs. IMO, it doesn't make sense
        #    to synthesize mov instructions.

        blacklist = [
            "VZEROALL",
            "VZEROUPPER",
            "SHLDL-R32-R32",
            "SHLDQ-R64-R64",
            "SHLDW-R16-R16",
            "SHRDL-R32-R32",
            "SHRDQ-R64-R64",
            "SHRDW-R16-R16",
        ]

        for inst in sorted(isa):
            logger.info(f"Processing: {inst}")
            sp = SimplParser(inst)
            try:
                sp.do_parse()
            except AssertionError as err:
                self.isa_stats["error"].append(inst)
                logger.error(f"Failed on: {inst}, {err}")
                continue
            except ValueError as err:
                self.isa_stats["error"].append(inst)
                logger.error(f"Failed on: {inst}, {err}")
                continue

            instsem = sp.instsem

            logger.info(
                f"{inst}:\n\tVec? {instsem.is_vec}\n\t"
                f"Float? {instsem.is_float}\n\t"
                f"R2? {'R2' not in instsem.sems}\n\t"
                f"mov? {instsem.opcode.startswith('mov')}\n\t"
                f"Ops: {len(instsem.operands)}\n\t"
                f"keys: {list(instsem.sems.keys())}"
            )

            if instsem.is_vec or instsem.is_float:
                self.isa_stats["float_or_vec"].append(inst)
                continue
            if instsem.name in blacklist:
                continue

            self.isa_stats["success"].append(inst)

            sp.infer_types()

            transformer = Transformer(sp.instsem)
            transformer.do_transforms()

            # TODO: Make this optional
            #rewriter = Rewriter()
            #rewriter.rewrite_inst(sp.instsem)

            sp.infer_types(normalized=True)
            self.register_inst(sp.instsem)

    def register_inst(self, instsem):
        self.semmap[instsem.name] = len(self.sems)
        self.sems.append(instsem)

    def filter_success(self, resultsf):
        with open(resultsf) as fd:
            results = yaml.load(fd)

        for key, res in results.items():
            if res.is_success():
                name = key.split("synth-target-")[1].rsplit("-", 1)[0]
                self.skiplist.append(name)

    def synth_workflow(self):
        fact = FACTORIZERS[self.factorizer]
        wm = WorkflowManager(self, fact)
        wm.process()

    # Helper Functions, called by other users of a SynthesisInstance, e.g., the workflow
    # manager.
    def prepare_synthesis_task(
        self,
        inst,
        component_set,
        force_include=[],
        timeout=3,
        max_prog_len=4,
        priority=10,
        pseudo=True,
        tags=[],
        target="REG",
        selector_name=None,
    ):

        if component_set is None:
            # Default to using the full ISA
            component_set = self.get_allowed_components(self.sems, inst)

        components = self.choose_components(inst, component_set, mode=target,
                                            selector_name=selector_name)
        pseudos = []
        if pseudo:
            if isinstance(pseudo, list):
                for pi in self.pseudo:
                    if pi.name in pseudo:
                        pseudos.append(pi)
            else:
                pseudos = self.pseudo

        components = components + force_include

        st = RosetteSynthesisTask(
            inst=inst,
            components=components,
            timeout=timeout,
            pseudo=pseudos,
            max_prog_len=max_prog_len,
            priority=priority,
        )

        # XXX: If factor name prefix changes, this needs to be adjusted
        if inst.name.startswith("pseudo"):
            st.add_tag("factor")

        for tag in tags:
            st.add_tag(tag)

        if target == "FLAG":
            st.flag_task = True

        st.selector_name = selector_name or self.selector
        st.forced_comps = force_include

        return st

    def choose_components(self, inst, component_set,
                          num_components=16, mode="REG", selector_name=None):
        # Don't do any selection if the num_components is less than the component set size
        if len(component_set) < num_components:
            return [c for c in component_set if c.name != inst.name]

        # Initialize the selector using the strategy
        selector_name = selector_name or self.selector
        synthesis_selector = selector[selector_name](component_set)

        comps = synthesis_selector.components_for(inst, count=num_components, mode=mode)
        logger.info(f"Using components (for {inst.name}): {[x.name for x in comps]}")
        return comps

    def get_allowed_components(self, sems, inst):
        cset = []
        added = set()
        set_has_inst = False
        task_name = inst.get_metadata("main_task", default="")

        for sem in sems:
            if sem.name in added:
                continue
            if sem.name == task_name:
                continue

            cset.append(sem)
            added.add(sem.name)

            if sem.name == inst.name:
                set_has_inst = True

        if not set_has_inst:
            cset.append(inst)

        return cset

    # Debug and Data Dump Functionality

    def _save_result(self, result, st):
        if self.result_queue:
            self.result_queue.put(json.dumps(result.__dict__))

        if self.conn:
            self.collection.insert_one(result.__dict__)

        self.status[result.task] = result
        #self.isa_stats["synth_err"].extend(st.isa_stats["synth_err"])

    def print_stats(self):
        # TODO: Make output directory configurable
        path = f"synth-results/synth-instance-{self.id}.txt"
        logger.info(f"Writing results to {path}")
        with open(path, "w") as fd:
            for k, result in self.status.items():
                fd.write(f"{k}:\n")
                fd.write(f"{result}\n")

        # Dump raw results to file as well
        with open(f"synth-results/synth-instance-{self.id}.yaml", "w") as fd:
            yaml.dump(self.status, fd)

        with open(f"synth-results/inst-parse-stats-{self.id}.yaml", "w") as fd:
            yaml.dump(dict(self.isa_stats), fd)


if __name__ == "__main__":
    argp = argparse.ArgumentParser()

    argp.add_argument(
        "--isa", nargs="+", type=str, help="Instruction part of current ISA"
    )

    argp.add_argument(
        "--only",
        type=str,
        nargs="*",
        default=[],
        help="Only synthesize instructions specified as args",
    )

    argp.add_argument(
        "--pseudo-inst",
        default=None,
        type=str,
        help="Load pseudo instructions to use in synthesis from yaml file",
    )

    argp.add_argument(
        "--selector",
        default="simple",
        type=str,
        choices=selector.keys(),
        help="Strategy to use for component selector",
    )

    argp.add_argument(
        "--redo-failures",
        type=str,
        help="Load a previous synthesis result dump and redo failures only",
    )

    argp.add_argument(
        "--parallel-tasks",
        type=int,
        default=1,
        help="Number of synthesis tasks to run in parallel",
    )

    argp.add_argument(
        "--try-instruction-factorization",
        action="store_true",
        help="Try instruction factorization if the synthesis fails",
    )

    argp.add_argument(
        "--factorizer",
        default="bottom_up",
        type=str,
        choices=FACTORIZERS.keys(),
        help="Strategy to use for instruction factorization",
    )

    argp.add_argument(
        "--iterative-rewrites",
        action='store_true',
        help="Enable iterative node-splitting for complex instructions",
    )

    argp.add_argument(
        "--timeout",
        type=int,
        default=20 * 60,
        help="Set per-synthesis task timeout in seconds",
    )

    argp.add_argument(
        "--synthesize-pseudo",
        action="store_true",
        help="Run synthesis for pseudo instructions only",
    )

    argp.add_argument(
        "--synthesize-flags",
        action="store_true",
        help="Run synthesis to fixup flags rather than output registers",
    )

    argp.add_argument(
        "--conn",
        type=str,
        default=None,
        help="Connection configuration file for various DBs and Queues"
    )

    argp.add_argument(
        "--long-run",
        action='store_true',
        help="Setup persistent/resilient storage options for long runs"
    )

    args = argp.parse_args()

    si = SynthesisInstance(
        args.isa,
        args.only,
        args.pseudo_inst,
        args.selector,
        redo_failures=args.redo_failures,
        parallel_tasks=args.parallel_tasks,
        try_factorization=args.try_instruction_factorization,
        factorizer=args.factorizer,
        timeout=args.timeout,
        synthesize_pseudo=args.synthesize_pseudo,
        synthesize_flags=args.synthesize_flags,
        conn=args.conn,
        long_run=args.long_run,
        iterative_rewrites=args.iterative_rewrites,
    )

    # This is long-running, catch ctrl+c to terminate gracefully and dump
    # partial results
    try:
        si.synth_workflow()
    except KeyboardInterrupt:
        logger.warning("Ctrl+C pressed, dumping partial results!")
    except AssertionError as err:
        logger.warning(f"Failed assertion in synthesis: {err}")
    finally:
        si.print_stats()
