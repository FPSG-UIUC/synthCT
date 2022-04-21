# Implements top-down factorization
from loguru import logger

from synthesis.task_result import SynthesisTaskResult
from synthesis.factorization import SubgoalsGenerator


class FactorizeTopDown(SubgoalsGenerator):
    def __init__(self, si, isa):
        self.isa = isa
        super().__init__(si)

    def generate_feedback(self, sem, st, history):
        # Iterate over set of possible decompositions for a single
        # instructions' ast
        # Ideally, we generate a new split at every operand to the top-level
        # k-opcode. But, there are a few exceptions:
        #   1. kextract: Only split the BV, not the bounds
        #   2. kif: Do not split the conditional. This is more of a limitation
        #   as the program will not typecheck otherwise.

        pot_roots = [sem.root]

        roots = []

        while pot_roots:
            cur = pot_roots.pop(0)
            logger.info(f"No split @ {cur} -> {sem.node_data(cur, 'no_split', False)}")
            if not sem.node_data(cur, "no_split", False):
                # Only consider this to be a root if there's at least one successor
                if len(list(sem.successors_ordered(cur))) > 0:
                    roots.append(cur)
                    break

            pot_roots.extend(list(sem.successors_ordered(cur)))

        for root in roots:
            succs = list(sem.successors_ordered(root))
            nd = sem.node_data(root, "data")

            if nd == "mi":
                succs = []
            elif nd == "extractMInt":
                succs = succs[:1]

            succs = [i for i in succs if not sem.node_data(i, "no_split", False)]

            for nidx in succs:
                yield nidx

    def factorize(self, st, flow, task_id, is_factor, parent):
        inst = st.spec
        factors = []
        factor_tasks = []

        sem = None
        reg = None
        for ireg, isem in inst.iter_output_regs():
            reg = ireg
            sem = isem

        for split_node_index in self.generate_feedback(sem, st, None):
            sinst, sinst_residual = inst.split_at_nidx(reg, split_node_index)

            # Check if either of the split instructions are "trivial". If it
            # is, we haven't made the problem any simpler and this is a bad
            # split. So ignore and continue.
            if self._is_inst_trivial(sinst, reg) or self._is_inst_trivial(
                sinst_residual, reg
            ):
                continue

            # If inst input to this function has no "factor" set then this is the main
            # synthesis task.
            # This is then successively copied down to smaller factors during subsequent
            # factorizations
            if st and "factor" not in st.tags:
                sinst.set_metadata("main_task", inst.name)
                sinst_residual.set_metadata("main_task", inst.name)
            elif st:
                if not sinst.get_metadata("main_task", default=False):
                    logger.error(
                        "Factor ({st.name}: {sinst.name}) with no `main_task`?!"
                    )
                if not sinst_residual.get_metadata("main_task", default=False):
                    logger.error(
                        "Factor ({st.name}: {sinst_residual.name}) with no `main_task`?!"
                    )

            factors.append([sinst, sinst_residual])

        # Iterate overthe pairs of factors that are generated,
        # Generate synthesis tasks for the factors,
        # Add then to the flow graph
        options = []
        alls = []
        for factor in factors:
            split, split_r = factor

            st = self.si.prepare_synthesis_task(
                split,
                component_set=None,
                max_prog_len=2,
                timeout=self.si.timeout,
                tags=["pseudo_only"],
            )

            st_r = self.si.prepare_synthesis_task(
                split_r,
                component_set=None,
                max_prog_len=2,
                timeout=self.si.timeout,
                tags=["pseudo_only"],
            )

            added, stid = flow.add_task_dedup(st.id, st)
            if not added:
                # The task we just tried to add is equivalent to some other existing task,
                # note the equivalence and save the result.
                equiv_name = flow.tasks[stid].spec.name
                fake_task_result = SynthesisTaskResult(
                    st, state="eq", name=st.id, data=equiv_name
                )
                self.si._save_result(fake_task_result, st)

            addedr, strid = flow.add_task_dedup(st_r.id, st_r)
            if not addedr:
                # Similar to above
                equiv_name = flow.tasks[strid].spec.name
                fake_task_result = SynthesisTaskResult(
                    st_r, state="eq", name=st_r.id, data=equiv_name
                )
                self.si._save_result(fake_task_result, st_r)

            flow.set_task_ready(stid)
            flow.set_task_ready(strid)

            allid = flow.add_all([stid, strid])
            flow.set_task_ready(allid)

            # Now, retry synthesis of the current instruction
            stc = self.si.prepare_synthesis_task(
                inst,
                [],
                force_include=[split, split_r],
                max_prog_len=4,
                timeout=self.si.timeout,
                pseudo=["PNOP", "PMOVQ-R64-R64", "MOVQ-IMM-R64"],
                tags=[],
            )

            flow.add_task(stc.id, stc)
            flow.set_task_ready(stc.id)

            flow.add_subtasks(stc.id, [allid])
            alls.append(allid)

            options.append(stc.id)

            factor_tasks.extend([stid, strid])

        anyid = flow.add_any(options)
        flow.set_task_ready(anyid)

        flow.add_subtasks(anyid, options)

        if is_factor:
            # Get parents
            this_parents = list(flow.get_parent_tasks(task_id))
            # Unlink the failed task
            flow.unlink_task(task_id)
            # Connect the parent tasks to the any
            for parent in this_parents:
                flow.add_subtasks(parent, [anyid])
            # Connect the failed task to the next tasks
            flow.add_subtasks(task_id, alls)

        return factor_tasks

    def feedback(self, st, flow, task_id, is_factor, parent):
        if flow.get_task_state(task_id) == "succ":
            return
        return self.factorize(st, flow, task_id, is_factor, parent)
