# Implements some basic transformations on semantics to make the ASTs more
# efficient for SMT based reasoning and synthesis.
#
# Also irons out some differences between semantics as implemented in
# rosette/SMTLIB and K-semantics.

from loguru import logger
import copy

from semantics.typing import Typing


class Transformer:
    def __init__(self, sem):
        self.sem = sem

    def do_transforms(
        self, names=["normalize_extracts", "normalize_shifts", "bitvec_conversion"]
    ):
        for lhs in self.sem.iter_semantics():
            for name in names:
                sem = self.sem.sems[lhs]
                if name == "normalize_extracts":
                    new_sem = self.normalize_extracts(sem)
                    Typing.infer(new_sem)
                    self.sem.sems[lhs] = new_sem
                if name == "normalize_shifts":
                    new_sem = self.normalize_shifts(sem)
                    Typing.infer(new_sem)
                    self.sem.sems[lhs] = new_sem
                if name == "bitvec_conversion":
                    new_sem = self.transform_bitvec_conversion(sem)
                    Typing.infer(new_sem)
                    self.sem.sems[lhs] = new_sem

    def normalize_extracts(self, _sem):
        sem = copy.deepcopy(_sem)
        for node in sem.sem.nodes():
            nty = sem.node_data(node, "data")
            if nty in ["extractMInt"]:
                bv, lower, upper = sem.successors_ordered(node)

                # Get the input size
                bvsz = sem.node_data(bv, Typing.KEY).sz

                new_low = bvsz - sem.node_data(lower, "data") - 1
                new_up = bvsz - sem.node_data(upper, "data")

                sem.replace_node_data(lower, new_up, "data")
                sem.replace_node_data(upper, new_low, "data")
        return sem

    def normalize_shifts(self, _sem):
        sem = copy.deepcopy(_sem)
        changed = True

        while changed:
            to_remove = list()
            changed = False
            for node in sem.sem.nodes():
                nty = sem.node_data(node, "data")
                if nty not in ["shiftLeftMInt", "aShiftRightMInt", "lshrMInt"]:
                    continue

                _, shift_amount = sem.successors_ordered(node)
                # Check if shift amount is a type conversion
                nty = sem.node_data(shift_amount, "data")
                if nty in ["svalueMInt", "uvalueMInt"]:
                    # Eliminate the type conversion
                    child = list(sem.successors_ordered(shift_amount))[0]
                    # Add an edge directly from node to child
                    sem.remove_edge(node, shift_amount)
                    sem.add_edge(node, child, 1)

                    to_remove.append(shift_amount)
                    changed = True

            for node in to_remove:
                sem.remove_node(node)

        return sem

    def transform_bitvec_conversion(self, sem):
        # Handles ASTs where bitvecs are converted from bitvec theory to
        # integer and back to bitvecs. The original and the final bitvec may
        # have different bitwidths, so the pass needs to emit additional bit
        # width change operations to ensure consistency.
        #
        # Such conversions, i.e., bitvec -> integer -> bitvec, causes the
        # solver to be inefficient. This conversion ensures better solver
        # performance.
        copy_sem = copy.deepcopy(sem)

        for node in sem.sem.nodes():
            nty = sem.node_data(node, "data")

            # The transformation is always only triggered the point of
            # creating a bitvec, e.g., `mi`
            if nty != "mi":
                continue

            width_idx, value_idx = sem.successors_ordered(node)

            # Check if either width_idx or value_idx are themselves trees of
            # depth > 1.
            width_data = sem.node_data(width_idx, "data")
            try:
                width_value = int(width_data)
            except ValueError:
                # For width, just emit a warning for now
                logger.warning("Width operand may be symbolic!")
                continue

            value_data = sem.node_data(value_idx, "data")
            value_is_int = True
            try:
                _ = int(value_data)
            except ValueError:
                value_is_int = False

            if value_is_int:
                continue

            if value_data not in ["svalueMInt", "uvalueMInt"]:
                continue

            # Ok, so now there's a type conversion from a bitvec -> integer
            # Add additional handling
            vsubtree = list(sem.successors_ordered(value_idx))[0]
            vsubtree_bwidth = sem.node_data(vsubtree, Typing.KEY).sz

            # Compare the original and resuling bv sizes
            if vsubtree_bwidth == width_value:
                # No need to change bitwidths, just a simple reroute
                reroute_to = vsubtree
            elif vsubtree_bwidth < width_value:
                # Do a sign/zero-extension based on if value_data was a
                # svalueMInt or a uvalueMInt
                # XXX: There is a bug here, sextMInt and uextMInt need a second operand!
                if value_data == "svalueMInt":
                    reroute_to = copy_sem.add_node(data="sextMInt")
                else:
                    reroute_to = copy_sem.add_node(data="uextMInt")
                widx = copy_sem.add_node(data=width_value)
                copy_sem.add_edge(reroute_to, vsubtree, 0)
                copy_sem.add_edge(reroute_to, widx, 1)
            else:
                # Emit an extract operation to limit the bitwidth size
                extract_idx = copy_sem.add_node(data="extractMInt")
                base = copy_sem.add_node(data=0)
                extent = copy_sem.add_node(data=width_value)

                copy_sem.add_edge(extract_idx, vsubtree, idx=0)
                copy_sem.add_edge(extract_idx, base, idx=1)
                copy_sem.add_edge(extract_idx, extent, idx=2)

                reroute_to = extract_idx

            # Do re-routing. Check if the current `node` has predecessors
            # Else, this is the new root.
            preds = list(copy_sem.predecessors(node))
            if preds:
                parent = preds[0]
                pdata = copy_sem.node_data(parent, "data")
                idx = copy_sem.edge_data(parent, node, key="idx")
                copy_sem.remove_edge(parent, node)
                copy_sem.add_edge(parent, reroute_to, idx=idx)

                # Cleanup/delete redundant nodes
                copy_sem.remove_node(width_idx)
                copy_sem.remove_node(value_idx)
                copy_sem.remove_node(node)
            else:
                # Cleanup/delete redundant nodes
                copy_sem.remove_node(width_idx)
                copy_sem.remove_node(value_idx)
                copy_sem.remove_node(node)

        return copy_sem
