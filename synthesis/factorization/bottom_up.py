# Implements Bottom-Up Factorization
from loguru import logger
from collections import defaultdict
import copy

from synthesis.task_result import SynthesisTaskResult
from synthesis.factorization import SubgoalsGenerator
from semantics.typing import Typing, BitVec


class FactorizeBottomUp(SubgoalsGenerator):
    def __init__(self, si, isa):
        self.isa = isa
        self.factor_store = []

        super().__init__(si)

    def factorize(self, st, flow, task_id, is_factor, parent):
        inst = st.spec
        outputs = [reg for reg, _ in inst.iter_output_regs()]
        mode = "REG"
        if st.flag_task:
            outputs = [flag for flag, _ in inst.iter_output_flags()]
            mode = "FLAG"

        leaves = []
        wl = []
        for reg in outputs:
            root = inst.sems[reg].root
            rt, factor = self.factorize_recursive(
                flow, inst, reg, root, root, self.isa, inst.name, mode=mode)

            if rt is not None and factor is not None:
                wl.append((rt, factor))

        # If this is not a flag task, add another task that connects all the output
        # registers
        if not st.flag_task and len(wl) > 1:
            all = flow.add_all([t for t, _ in wl])
            flow.set_task_ready(all)

            st = self.si.prepare_synthesis_task(
                inst,
                component_set=[],
                timeout=self.si.timeout,
                max_prog_len=1,
                force_include=[i for _, i in wl],
                target=mode)

            flow.add_task(st.id, st)
            flow.add_subtasks(st.id, [all])
            flow.set_task_ready(st.id)

            wl = [st.id]
        else:
            wl = [t for t, _ in wl]

        while wl:
            task = wl.pop(0)
            subtasks = list(flow.iter_subtasks(task))
            if not subtasks:
                leaves.append(task)
            else:
                wl.extend(subtasks)

        # Try generalization to reduce the number of tasks

        # factors = []
        # tnames = dict()
        # for taskid, task in flow.tasks.items():
        # finst = task.spec
        # factors.append(finst)
        # tnames[finst.name] = taskid

        # gen = GoalGeneralizer()
        # ginsts, eqsets = gen.generalize(factors, 'R2')

        # for hash, insts in eqsets.items():
        # if len(insts) < 2:
        # continue

        # ginst = ginsts[hash][0]
        # st = self.prepare_synthesis_task(
        # ginst,
        # self.get_allowed_components(self.isa, ginst),
        # max_prog_len=2,
        # timeout=self.timeout)

        # stid = st.id
        # added, stid = flow.add_task_dedup(stid, st)
        # flow.set_task_ready(stid)

        # for inst in insts:
        # tid = tnames[inst.name]

        ## Create a new task for inst with ginst as the only component
        # newst = self.prepare_synthesis_task(
        # inst,
        # [],
        # force_include=[ginst],
        # max_prog_len=2,
        # timeout=self.timeout)

        # newstid = newst.id
        # flow.add_task(newstid, newst)
        # flow.set_task_ready(newstid)

        # flow.add_subtasks(newstid, [stid])

        # parents = flow.get_parent_tasks(tid)
        # for parent in parents:
        # if not parent.startswith("ANY"):
        # flow.remove_subtask(parent, tid)
        # anyid = flow.add_any([tid, newstid])
        # flow.set_task_ready(anyid)
        # flow.add_subtasks(parent, [anyid])
        # else:
        # flow.add_subtasks(parent, [newstid])

        # if tid in leaves:
        # del leaves[leaves.index(tid)]

        # leaves.append(stid)

        return leaves

    def factorize_recursive(self, flow, inst, reg, iroot, nidx, isa, main_task, mode):
        sem = inst.sems[reg]
        succs = list(sem.successors_ordered(nidx))
        if not list(succs):
            return None, None

        succ_tasks = []
        include = []

        if nidx != iroot:
            cur_inst, _ = inst.split_at_nidx(reg, nidx)
            cur_inst.set_metadata("main_task", main_task)
        else:
            cur_inst = inst

        for succ in succs:
            subtask, subinst = self.factorize_recursive(
                flow, inst, reg, iroot, succ, isa, main_task, mode,
            )
            if subtask is not None:
                _, cur_inst = cur_inst.split_at_nidx(reg, succ)
                cur_inst.set_metadata("main_task", main_task)
                include.append(subinst)
                succ_tasks.append(subtask)

        st = self.si.prepare_synthesis_task(
            cur_inst,
            component_set=[],
            force_include=include,
            max_prog_len=1,
            timeout=self.si.timeout,
            target=mode,
            tags=["pseudo_only"],
        )

        # Save the factor for future use.
        self.factor_store.append(cur_inst)

        stid = st.id
        added, stid = flow.add_task_dedup(stid, st)
        if not added:
            # The task we just tried to add is equivalent to some other existing task,
            # note the equivalence and save the result.
            equiv_name = flow.tasks[stid].spec.name
            fake_task_result = SynthesisTaskResult(
                st, state="eq", name=st.id, data=equiv_name
            )
            self.si._save_result(fake_task_result, st)

        # XXX: This may cause a bug if we change the implementation to dispatch tasks
        # eagerly, i.e., before the whole graph is built, later on.
        flow.set_task_ready(stid)
        tid = stid

        # If this task was broken down into an "ALL", add a parent for the actual task
        if succ_tasks:
            logger.info(f"Adding ALL: {succ_tasks} {stid}")
            tid = flow.add_all([*succ_tasks, stid])
            flow.set_task_ready(tid)

            this_inst = None
            if nidx != iroot:
                this_inst, _ = inst.split_at_nidx(reg, nidx)
            else:
                this_inst = inst.clone_with_keys([reg])

            this_inst.set_metadata("main_task", main_task)
            st = self.si.prepare_synthesis_task(
                this_inst,
                [],
                force_include=[*include, cur_inst],
                max_prog_len=1,
                timeout=self.si.timeout,
                target=mode,
                pseudo=["PNOP", "PMOVQ-R64-R64", "MOVQ-IMM-R64"],
            )

            self.factor_store.append(this_inst)

            pid = st.id
            flow.add_task(pid, st)
            flow.add_subtasks(pid, [tid])
            flow.set_task_ready(pid)

            # Add another task for the inst, but with all components
            st2 = self.si.prepare_synthesis_task(
                this_inst,
                component_set=None,
                timeout=self.si.timeout,
                max_prog_len=1,
                target=mode,
            )

            flow.add_task(st2.id, st2)
            flow.set_task_ready(st2.id)

            logger.info(f"Added granular task: {st2.id}: {st2.name} {main_task}")

            # Create an "ANY" Node
            anyid = flow.add_any([pid, st2.id])
            flow.set_task_ready(anyid)

            tid = anyid
            cur_inst = this_inst

        return tid, cur_inst

    def feedback(self, st, flow, task_id, is_factor, parent):
        # In bottom-up, all factors are generated at one shot and updated into the
        # workflow graph. Therefore, the actual feedback step is a noop.
        if "factor" in st.tags:
            return []
        return


class GoalGeneralizer:
    def __init__(self):
        pass

    def generalize(self, insts, reg):
        hashes = defaultdict(list)
        eq_sets = defaultdict(list)

        for inst in insts:
            generalized_inst = self.generalize_constants(inst, reg)
            if not generalized_inst:
                continue

            ghash = self.hash_inst(generalized_inst, reg)
            hashes[ghash].append(generalized_inst)

            eq_sets[ghash].append(inst)

        return hashes, eq_sets

    def generalize_constants(self, inst, reg):
        constants = defaultdict(list)

        newi = copy.deepcopy(inst)
        if newi.name.startswith("pseudo"):
            newi.name = f"pseudo-genc-{newi.name[7:]}"
        else:
            newi.name = f"genc-{newi.name}"

        sem = newi.sems[reg]

        for node in sem.sem.nodes():
            data = sem.node_data(node, key="data")
            value = None
            try:
                value = int(data)
            except ValueError:
                pass

            if value:
                # Some constants may not make sense to generalize
                parent = list(sem.predecessors(node))
                if not parent:
                    logger.warning("Constant has no parent?")
                    continue
                parent = parent[0]
                pdata = sem.node_data(parent, key="data")
                eidx = sem.edge_data(parent, node, key="idx")

                if pdata in ["zextMInt"]:
                    continue
                if pdata == "mi" and eidx == 0:
                    # Do not generalize the bitwidth of a BV
                    continue

                constants[value].append(node)

        # We have all the constants, replace them with registers,
        # TODO: making sure that the types are consistent
        used_regs = sem._used_regs_in_sem()
        available = [f"R{i}" for i in range(1, 6)]

        for regr in used_regs:
            if regr in available:
                del available[used_regs.index(regr)]

        assert available, "No available registers?"

        if len(constants) > 2:
            logger.info("Too may constants. Avoiding generalization!")
            return None

        for const, nodes in constants.items():
            areg = available.pop(0)
            for nidx in nodes:
                parent = list(sem.predecessors(nidx))
                if not parent:
                    logger.warning("Constant has no parent in AST?")
                    continue

                parent = parent[0]
                newn = sem.add_node(data=areg)
                sem.add_node_data(newn, Typing.KEY, BitVec(64))
                idx = sem.edge_data(parent, nidx, key="idx")
                sem.remove_node(nidx)
                sem.add_operand_type_checked(parent, newn, idx=idx)

        newi._fixup_metadata(reg, None, False)

        return newi

    def hash_inst(self, inst, reg):
        return inst.sems[reg].hash()
