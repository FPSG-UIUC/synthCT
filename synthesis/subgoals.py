# This module implements classes and methods to divide a synthesis task into multiple
# sub-goals.
#
# Additionally, the sub-goals generated should also exploit memoization opportunities:
# both within a single instruction and across all synthesis targets in the current
# instruction instance. To maximize overlap and reduce the number of synthesis tasks to
# run, the generator should also have a global view of the synthesis instance.

from collections import defaultdict
import copy
import re
import itertools

from loguru import logger
import networkx as nx
import queue
import multiprocessing as mp

from synthesis.synth_task import spawn_synthesis_worker
from synthesis.verify_solution import FlagVerifier
from synthesis.rewrites.iterative import IterativeRewriteDivide

MAX_TIMEOUT = 3600 * 24
#MAX_TIMEOUT = 20


class WorkflowGraph:
    def __init__(self):
        self._idx = 0
        self.workflow = nx.DiGraph()

        self.mapping = dict()
        self.rmap = dict()

        self.state = dict()
        self.tasks = dict()

        self.hashes = defaultdict(list)
        self.is_factorized = False

    def is_empty(self):
        return self._idx == 0

    def add_node(self, id):
        cur = self._idx
        self._idx += 1

        self.workflow.add_node(cur)
        self.mapping[cur] = id
        self.rmap[id] = cur

        return cur

    def add_any(self, tasks):
        hash = ",".join(tasks)
        name = f"ANY-{hash}"

        _ = self.add_node(name)

        self.add_subtasks(name, tasks)

        return name

    def add_all(self, tasks):
        hash = ",".join(tasks)
        name = f"ALL-{hash}"

        _ = self.add_node(name)

        self.add_subtasks(name, tasks)

        return name

    def add_task_dedup(self, task_id, task):
        # This function adds a task if the task tree has not been synthesized already,
        # i.e., it first computes a hash of the AST, compares against existing known tasks
        # and adds to the task list only if the hash is unique. Otherwise, a prev. added
        # task id is returned instead.
        # Function returns a tuple: Bool, task_id; True indicates a new task was addded,
        # while False indicates a previously seen task is equivalent.
        task_hash = task.spec.generate_hash()
        if task_hash not in self.hashes:
            self.hashes[task_hash].append(task_id)
            self.add_task(task_id, task)
            return True, task_id

        reuse = self.hashes[task_hash][0]
        self.hashes[task_hash].append(task_id)

        logger.info(f"Found duplicate {task_hash} {task.spec.name} {reuse}")
        return False, reuse

    def add_task(self, task_id, task):
        logger.info(f"Added task {task_id}: {task.spec.name}")
        self.tasks[task_id] = task
        return self.add_node(task_id)

    def add_subtasks(self, parent, tasks):
        pidx = self.rmap[parent]
        for task in tasks:
            sidx = self.rmap[task]
            if sidx == pidx:
                logger.error(f"ADDING SELF-LOOP IN WORKFLOW! {parent} {task}")
                assert False
            self.workflow.add_edge(pidx, sidx)

    def unlink_task(self, task):
        nidx = self.rmap[task]
        for pidx in list(self.workflow.predecessors(nidx)):
            self.workflow.remove_edge(pidx, nidx)

    def remove_subtask(self, parent, task):
        pidx = self.rmap[parent]
        tidx = self.rmap[task]
        self.workflow.remove_edge(pidx, tidx)
        logger.info(f"Removing subtask {parent} {task}")

    def set_task_state(self, task_id, state):
        assert state in ["ready", "sched", "succ", "fail"], "Invalid task_state"
        self.state[task_id] = state

    def set_task_ready(self, task_id):
        self.set_task_state(task_id, "ready")

    def set_task_sched(self, task_id):
        self.set_task_state(task_id, "sched")

    def set_task_succ(self, task_id):
        self.set_task_state(task_id, "succ")

    def set_task_fail(self, task_id):
        self.set_task_state(task_id, "fail")

    def get_task_state(self, task_id):
        return self.state[task_id]

    def iter_subtasks(self, task_id):
        idx = self.rmap[task_id]
        for sidx in self.workflow.successors(idx):
            yield self.mapping[sidx]

    def get_parent_tasks(self, task_id):
        idx = self.rmap[task_id]
        for pidx in self.workflow.predecessors(idx):
            yield self.mapping[pidx]

    def debug_dump(self, fname=None):
        import tempfile as tf
        from matplotlib import pyplot as plt
        from networkx.drawing.nx_agraph import graphviz_layout

        if not fname:
            fname = tf.NamedTemporaryFile(suffix=".svg", delete=False).name

        plt.figure(3, figsize=(12, 9))
        pos = graphviz_layout(self.workflow, prog="dot", root=0, args="-Gsize=12,9")
        nx.draw(self.workflow, pos, node_color="w", node_shape="s")

        node_labels = dict()
        for k in self.workflow.nodes():
            tid = self.mapping[k]
            if tid.startswith("ANY"):
                node_labels[k] = "ANY"
            elif tid.startswith("ALL"):
                node_labels[k] = "ALL"
            else:
                node_labels[k] = f"{tid}"

        nx.draw_networkx_labels(self.workflow, pos, labels=node_labels, font_size=10)
        plt.savefig(fname)
        plt.show()
        plt.close()
        logger.success(f"Saved workflow graph to: {fname}")


class WorkflowManager:
    def __init__(self, si, fact):
        # Warning: Do not modify SI directly. Use only the API functions
        self.si = si
        self.flows = defaultdict(WorkflowGraph)
        self.factorizer = fact(si, si.sems)

        self.iteratives = []

        self.seen_tasks = set()

    def process(self):
        si = self.si

        procs = []
        for _ in range(si.parallel_tasks):
            wq = None
            if si.conn:
                wq = si.conn
            else:
                wq = si.wq
            p = mp.Process(
                target=spawn_synthesis_worker,
                args=(
                    wq,
                    si.result_q,
                ),
            )
            p.start()
            procs.append(p)

        # Populate the tasks in self.wq
        for name in si.local_wq:
            # Check if we're trying to synthesize a single factor
            # These are indicated by the name of the instruction starting with
            # a `pseudo-` prefix.
            if name.startswith("pseudo-"):
                name_parts = name.split("-")[1:]
                factor_parts = []
                realname = []
                for part in name_parts:
                    # Parse until we start to process the factor string
                    # part of the instruction name
                    matches = re.match(r"s(r)?([0-9]+)", part)
                    if matches:
                        residual = 0
                        if matches.group(1):
                            residual = 1
                        factor_parts.append((int(matches.group(2)), residual))
                    else:
                        realname.append(part)

                realname = "-".join(realname)
                realname, reg = realname.rsplit("_", 1)

                idx = si.semmap[realname]
                inst = si.sems[idx]

                inst = self.generate_factor(inst, reg, factor_parts)
            else:
                idx = si.semmap[name]
                inst = si.sems[idx]

            if inst.name in si.skiplist:
                logger.success(
                    f"Found {inst.name} as success in {si.redo_failures}." "Skipping!"
                )
                continue

            if inst.opcode.startswith("mov") or inst.opcode.startswith("cmov"):
                # Skip synthesis of mov and cmov
                # Remain in synth, remove from goals
                continue

            # Check if the inst even has any register outputs
            target = "REG"
            should_synth_flags = si.synthesize_flags or not list(inst.iter_output_regs())

            if should_synth_flags:
                if not list(inst.iter_output_flags()):
                    continue
                target = "FLAG"

            has_been_rewritten = False
            if si.iterative_rewrites:
                if IterativeRewriteDivide.can_rewrite(inst):
                    logger.debug(f"Can rewrite inst: {inst.name}")
                    irewriter = IterativeRewriteDivide(inst)
                    sinsts = irewriter.create_splits()
                    self.iteratives.append(irewriter)

                    for sinst in sinsts:
                        # TODO: This line is sus
                        si.sems.append(sinst)
                        st = self.si.prepare_synthesis_task(
                            inst=sinst,
                            component_set=None,
                            timeout=2, # TODO
                            max_prog_len=4,  # TODO
                            #timeout=si.timeout,
                            #max_prog_len=1,
                            tags=['iterative'],
                            target=target,
                        )
                        self._try_add_task(st)

                    has_been_rewritten = True

            if not has_been_rewritten:
                st = self.si.prepare_synthesis_task(
                    inst=inst,
                    component_set=None,
                    timeout=si.timeout,
                    max_prog_len=1,
                    tags=[],
                    target=target,
                )
                self._try_add_task(st)

        if si.synthesize_pseudo:
            tasks = self.synthesize_pseudo_instructions()
            for task in tasks:
                self._try_add_task(task)

        try:
            while True:
                # Stream results back from the results queue
                try:
                    result, st = si.result_q.get()
                except queue.Empty:
                    logger.warning("No new synthesis results. Shutting down.")
                    break

                logger.info(f"Result ({st.name}): {result}")
                self.process_feedback(result, st)
                si._save_result(result, st)
        except KeyboardInterrupt:
            # Cleanup all children
            for p in procs:
                if p.is_alive():
                    p.join(1)
            raise KeyboardInterrupt

        # Cleanup all children
        for p in procs:
            if p.is_alive():
                p.join(1)

    def generate_factor(self, inst, reg, factor_parts):
        for part in factor_parts:
            i, ir = inst.split_at_nidx(reg, part[0])
            if part[1] == 1:
                inst = ir
            else:
                inst = i
        inst.set_metadata("main_task", inst.name)
        return inst

    def synthesize_pseudo_instructions(self):
        tasks = []
        compnames = [
            "MOVQ-R64-R64",
            "NOTQ-R64",
            "SHLQ-R64-CL",
            "SHRQ-R64-CL",
            "ANDQ-R64-R64",
            "ADDQ-R64-R64",
            "SUBQ-R64-R64",
            "ORQ-R64-R64",
            "XORQ-R64-R64",
            "CMOVEQ-R64-R64",
        ]

        complist = []

        for inst in self.si.sems:
            if inst.name in compnames:
                complist.append(inst)

        for inst in self.si.pseudo:
            st = self.si.prepare_synthesis_task(
                inst,
                [],
                force_include=complist,
                timeout=self.si.timeout,
                pseudo=["MOVQ-IMM-R64", "PNOP", "PMOVQ-R64-R64"],
                max_prog_len=1,
                tags=["pseudo_inst"],
            )

            tasks.append(st)

        return tasks

    def process_feedback(self, result, st):
        is_factorization_enabled = self.si.try_factorization

        tid = st.id
        key = st.spec.name

        if "skip_feedback" in st.tags:
            logger.info(f"Completed {st.name}, no feedback!")
            return

        # Check if the current feedback is for a factor
        is_factor = is_factorization_enabled and "factor" in st.tags

        # Check if the task is for a factor and get the main instruction's name as the key
        # if it is
        parent_inst = None
        if is_factor:
            inst = st.spec.get_metadata("main_task", None)
            assert inst, f"No `main_task` for a factor? {st.name}"
            key = inst

            for instsem in self.si.sems:
                if instsem.name == inst:
                    parent_inst = instsem
                    break

        if is_factor and parent_inst is None:
            logger.error("Could not find parent instruction for factor?")
            return

        # Check if we're doing reg or flag synthesis
        if st.flag_task:
            key = f"{key}-FLAG"
        else:
            key = f"{key}-REG"

        flow = self.flows[key]
        if flow.is_empty():
            flow = None

        should_generate_multiple_solutions = (
            "pseudo_inst" not in st.tags
            and not is_factor  # noqa: W503
            and flow is None  # noqa: W503
        )

        is_iterative = 'iterative' in st.tags

        feedback_tasks = []

        if result.is_success():
            # We have a success, yay!
            if is_iterative:
                main = st.spec.get_metadata("main_task", None)
                for irwr in self.iteratives:
                    if irwr.name != main:
                        continue
                    irwr.on_success(st, result)

                    if irwr.is_done():
                        # TODO:
                        # Stitch the solution
                        # Append the solutions to our store
                        pass
                return
            # If this is a flag solution, skip this step
            elif not result.is_flag_result:
                factors = self.factorizer.factor_store
                fv = FlagVerifier(self.si.sems, self.si.pseudo)
                are_flags_correct = fv.verify_solution(result, factors=factors)
                result.flags_verified = are_flags_correct

                # Launch tasks to synthesize flags
                if not are_flags_correct:
                    # Its pointless to synthesize flags for a solution that has a
                    # blacklisted set of components if the flags were to use the whole
                    # component set, as the final solution, i.e., flag + reg will
                    # invariably include the blacklisted set of components.
                    #
                    # OTOH, the parent successes, the ones that set up a component in
                    # blacklist, will still be able to explore a flag solution that did
                    # not contain the now avoided component, therefore, we maintain
                    # "completeness"

                    bl = st.blacklist_comps
                    fst = self.si.prepare_synthesis_task(
                        inst=st.spec,
                        component_set=[i for i in self.si.sems if i not in bl],
                        timeout=self.si.timeout,
                        max_prog_len=1,
                        tags=[],
                        target="FLAG",
                    )
                    fst.flag_task = True

                    # Propagate the blacklist
                    fst.blacklist_comps = bl

                    feedback_tasks.append(fst)
                    logger.info(f"{result.name} pushed flag tasks to synth!")
                else:
                    logger.info(f"{result.name} has flags set correctly!")

            # Check if flags are set correctly (if needed)
            if should_generate_multiple_solutions:
                new_tasks = self.feedback_new_solution(st, result)
                feedback_tasks.extend(new_tasks)
        elif result.is_unsat():
            # If the result is an unsat, increase program length till we
            # get to the max program length
            tid = st.id
            plen = st.max_prog_len
            comp_count = len(st.components) or 1
            eligible_components = self._get_eligible_component_set(st)

            if plen < self.si.max_prog_len:
                st.max_prog_len += 1
                logger.info(f"[{st.name}] Increased len to: {st.max_prog_len}")
                feedback_tasks.append(st)
            else:
                selector_name = "knn"
                feedback = True
                if "pseudo_only" in st.tags:
                    # Do proper comonent selection
                    st.tags.remove("pseudo_only")
                    comp_count = max(4, len(st.forced_comps))
                    logger.info(f"[{st.name}] Trying with {comp_count} components")
                elif comp_count > 7 and "alternate" not in st.tags:
                    selector_name = "jaccard"
                    st.tags.add("alternate")
                    logger.info(
                        f"[{st.name}] Trying with alternate component seclector ({comp_count})"
                    )
                elif 2 * comp_count < self.si.max_comp_count:
                    comp_count = 2 * comp_count
                    logger.info(
                        f"[{st.name}] Trying with more ({comp_count}) components"
                    )
                else:
                    feedback = False

                if feedback:
                    mode = "FLAG" if st.flag_task else "REG"
                    components = self.si.choose_components(
                        inst=st.spec,
                        component_set=eligible_components,
                        num_components=comp_count,
                        selector_name=selector_name,
                        mode=mode,
                    )

                    st.components = [
                        comp
                        for comp in itertools.islice(
                            itertools.chain(st.forced_comps, components), comp_count
                        )
                    ]

                    # Reset program length and timeout to initial values
                    st.max_prog_len = 1
                    st.timeout = self.si.timeout

                    feedback_tasks.append(st)

            # If there was any feedback, then the task id is reused. Therefore, remove it
            # from seen_tasks to be able to requeue the task again.
            if feedback_tasks:
                # Remove it from seen tasks to be able to re-queue it
                self.seen_tasks.remove(tid)

        elif result.is_timeout():
            tid = st.id
            timeout = st.timeout * 2
            if timeout < 1:
            #if timeout < MAX_TIMEOUT:
                st.timeout = timeout
                # Remove it from seen tasks to be able re-queue it
                self.seen_tasks.remove(tid)
                logger.info(f"[{st.name}] Increased timeout to: {timeout}")
                feedback_tasks.append(st)

        # Push all the feedback tasks
        if feedback_tasks:
            for new_task in feedback_tasks:
                self._try_add_task(new_task, flow)
            return

        # If we get to this point, it means that we've exhausted all possibilities with
        # synthesizing an instruction as a whole and we're about to try factorization.
        # The following part of code should only be triggered if factorization is enabled.
        # XXX: Break the code below into a separate function later on for hygiene
        if not is_factorization_enabled:
            return

        # If this is the first time we're trigerring the feedback mechansim, gather the
        # "base" jobs directly from the factorizer, rather than from the updated graph --
        # which will serve as a feedback from the next iteration

        flow = self.flows[key]
        if flow.is_empty():
            new_tasks = self.factorizer.factorize(
                st, flow, task_id=tid, is_factor=is_factor, parent=parent_inst
            )

            for task_id in new_tasks:
                task = flow.tasks[task_id]
                self._try_add_task(task, flow)

            return

        # If the program gets here, it means we've tried everything with existing task, or
        # it has succeeded. Time to try out new tasks.
        if result.is_success():
            flow.set_task_succ(tid)

        # Allow the registered factorizer to create feedback
        self.factorizer.feedback(
            st, flow, task_id=tid, is_factor=is_factor, parent=parent_inst
        )

        state = flow.get_task_state(tid)

        if state == "succ":
            self.process_feedback_success(flow, tid)
        else:
            self.process_feedback_failure(flow, tid)

    def _get_eligible_component_set(self, st):
        spec_name = st.spec.name
        is_factor = "factor" in st.tags
        component_set = []
        bl = st.blacklist_comps

        if is_factor:
            parent_inst = st.spec.get_metadata("main_task", None)
            if parent_inst not in bl:
                bl.append(parent_inst)
            for factor in self.factorizer.factor_store:
                if factor.name == spec_name:
                    component_set.append(factor)
                    break

        component_set.extend([i for i in self.si.sems if i.name not in bl])
        # component_set.extend([i for i in self.si.sems if i not in bl])

        return component_set

    def _try_add_task(self, task, flow=None):
        tid = task.id
        if tid in self.seen_tasks:
            logger.warning(f"Dropping {tid}; found duplicate!")
            return
        self.seen_tasks.add(tid)
        self.si.wq.put(task, priority=task.priority)
        logger.info(f"Sent task to worker: {task.spec.name}")
        if flow:
            flow.set_task_sched(tid)

    def feedback_new_solution(self, st, result):
        # st is success and we have a synthesis solution
        # Its time to try to introduce randomness and generate more than one solution
        inst = st.spec
        components = st.components
        cbl = st.blacklist_comps

        prog = result.program.split("\n")
        opcodes = []

        logger.info(f"Creating new tasks: {st.name}: {prog}")

        for line in prog[1:]:
            opcode = line.split(" ")[0]
            if opcode not in opcodes:
                opcodes.append(opcode)

        new_sets = []
        useds = []

        for idx, comp in enumerate(components):
            if comp.name.lower() in opcodes:
                newcomps = copy.deepcopy(components)
                del newcomps[idx]
                new_sets.append(newcomps)
                useds.append(components[idx])

        new_tasks = []
        prog_len = st.max_prog_len

        for used_comp, set in zip(useds, new_sets):
            avoid = cbl + [used_comp]
            target = "REG" if not st.flag_task else "FLAG"
            nst = self.si.prepare_synthesis_task(
                inst,
                [],
                force_include=set,
                max_prog_len=prog_len,
                timeout=st.timeout,
                target=target,
            )
            nst.blacklist_comps = avoid

            # If the success is from a flag task, set the feedback to be a flag task as
            # well
            if st.flag_task:
                nst.flag_task = True
            new_tasks.append(nst)

        return new_tasks

    def process_feedback_success(self, flow, task_id):
        # Get the parent task
        ptaskids = flow.get_parent_tasks(task_id)

        # No parent, NOOP
        if not ptaskids:
            return

        for ptaskid in ptaskids:
            # Meta task, process them
            if ptaskid.startswith("ANY"):
                # If the parent is an "ANY", mark the any node as a success and recurse
                flow.set_task_succ(ptaskid)
                self.process_feedback_success(flow, ptaskid)
                # TODO: Revoke any scheduled/running subtasks of ANY
            elif ptaskid.startswith("ALL"):
                # If the parent is an "ALL", check all the children.
                # If all the children are "succ", then, recurse to parent, else wait.
                is_all_succ = True
                for subtaskid in flow.iter_subtasks(ptaskid):
                    subtask_state = flow.get_task_state(subtaskid)
                    if subtask_state != "succ":
                        is_all_succ = False
                        break

                if not is_all_succ:
                    continue

                flow.set_task_succ(ptaskid)
                self.process_feedback_success(flow, ptaskid)
            else:
                # parent is a normal task, process it
                state = flow.get_task_state(ptaskid)
                if state != "ready":
                    continue

                ptask = flow.tasks[ptaskid]
                self._try_add_task(ptask, flow)

    def process_feedback_failure(self, flow, task_id, depth=0):
        has_all_subtasks_failed = True
        has_any_subtask_failed = False

        for subtaskid in flow.iter_subtasks(task_id):
            state = flow.get_task_state(subtaskid)
            # This task is in some state other than ready, don't really care what state,
            # i.e., either success or failure, but we do not need to process it
            if state == "fail":
                has_any_subtask_failed = True
                continue
            elif state == "succ" or state == "sched":
                has_all_subtasks_failed = False
                continue

            # If execution reaches here, it means that the task is in ready state
            has_all_subtasks_failed = False

            # Ok, the task is ready, check if this is a "meta" task
            if subtaskid.startswith("ANY") or subtaskid.startswith("ALL"):
                # TODO: Revoke/remove tasks if all/any subtasks of meta-tasks fail
                self.process_feedback_failure(flow, subtaskid, depth=depth + 1)
                continue

            # An ordinary task that is ready, try to schedule it
            subtask = flow.tasks[subtaskid]
            self._try_add_task(subtask, flow)

        has_current_task_failed = False
        if not task_id.startswith("ANY") and not task_id.startswith("ALL"):
            has_current_task_failed = True
        elif task_id.startswith("ANY") and has_all_subtasks_failed:
            flow.set_task_fail(task_id)
            has_current_task_failed = True
        elif task_id.startswith("ALL") and has_any_subtask_failed:
            flow.set_task_fail(task_id)
            has_current_task_failed = True

        if has_current_task_failed:
            flow.set_task_fail(task_id)
            if depth > 0:
                return
            for ptaskid in flow.get_parent_tasks(task_id):
                self.process_feedback_failure(flow, ptaskid, depth=depth - 1)
