# Implements SynthesisGraph, a data-structure that encodes results of multiple
# synthesis tasks

import argparse
import tempfile as tf
from collections import defaultdict
import re
import uuid
import itertools

from loguru import logger

import numpy as np
import pandas as pd


import networkx as nx
import yaml

from synthesis.work_queue import BeanstalkWQ, ListWQ
from synthesis.task_result import SynthesisTaskResult


class SynthesisGraph:
    # Define some colors
    unsafe_color = "red"
    safe_color = "green"
    success_color = "cadetblue2"

    def __init__(self):
        self.sg = nx.MultiDiGraph()
        self.node_map = dict()
        self.ctr = -1
        self.stats = defaultdict(lambda: 0)
        self.programs = defaultdict(list)

    def add_node(self, opcode):
        opcode = opcode.upper()
        if opcode in self.node_map:
            return self.node_map[opcode]

        self.ctr += 1

        self.sg.add_node(self.ctr, label=opcode, data=opcode)
        self.node_map[opcode] = self.ctr

        return self.ctr

    def set_node_safe(self, node):
        self.sg.nodes[node]["state"] = "safe"

    def set_node_unsafe(self, node):
        self.sg.nodes[node]["state"] = "unsafe"

    def set_node_synthesizeable(self, node):
        self.sg.nodes[node]["state"] = "synth"

    def iter_nodes(self):
        return self.sg.nodes()

    def iter_solutions(self, node):
        solutions = set()
        for edge in self.sg.out_edges(node, data=True):
            solutions.add(edge[2]["tag"])

        return list(solutions)

    def iter_solution(self, node, solution):
        nexts = []
        for edge in self.sg.out_edges(node, data=True):
            if edge[2]["tag"] != solution:
                continue
            nexts.append((edge[1], edge[2]["lineno"], edge[2]["ops"]))

        return [(x[0], x[2]) for x in sorted(nexts, key=lambda y: y[1])]

    def discard_solution(self, node, solution):
        edges_to_remove = []

        for edge in self.sg.edges(node, keys=True, data=True):
            if edge[3]["tag"] == solution:
                edges_to_remove.append((edge[0], edge[1], edge[2]))

        for p, c, i in edges_to_remove:
            self.sg.remove_edge(p, c, key=i)

    def iter_users(self, node):
        return [x[0] for x in self.sg.in_edges(node)]

    def inst(self, node):
        return self.sg.nodes[node]["data"]

    def get_safe_sets(self):
        from sympy import symbols
        from functools import reduce
        import operator
        from sympy.logic import simplify_logic
        from sympy.logic.boolalg import to_cnf, to_dnf

        vars = dict()
        expressions = dict()
        for node in self.sg.nodes():
            vars[node] = symbols(self.inst(node))

        for node in self.sg.nodes():
            exprs = []
            for sol in self.iter_solutions(node):
                expr = None
                for nidx, _ in self.iter_solution(node, sol):
                    if expr:
                        expr = expr & vars[nidx]
                    else:
                        expr = vars[nidx]
                exprs.append(expr)
            exprs.append(vars[node])
            expressions[node] = reduce(operator.or_, exprs)

        final_expression = reduce(operator.and_, expressions.values())
        print(to_dnf(final_expression, simplify=True, force=True))

    def get_safe_sets_nnf(self):
        from nnf import Var, Or, And

        big_vars = dict()
        small_vars = dict()
        expressions = dict()

        source_node = None
        for node in self.sg.nodes():
            inst = self.inst(node)
            if inst == 'ORQ-R64-R64':
                source_node = node
                break

        small_graph = nx.induced_subgraph(
            self.sg,
            [x for x in nx.dfs_tree(self.sg, source=source_node, depth_limit=2)]).copy()
        self.sg = small_graph
        self.to_svg("SMALL-GRAPH.svg")

        for node in self.sg.nodes():
            big_vars[node] = Var(self.inst(node))
            small_vars[node] = Var(self.inst(node).lower())

        for node in self.sg.nodes():
            exprs = []
            for sol in self.iter_solutions(node):
                expr = And([big_vars[nidx] for nidx, _ in self.iter_solution(node, sol)])
                exprs.append(expr)

            exprs.append(small_vars[node])
            expressions[node] = Or(exprs)

        final_expression = And(expressions.values())

        logger.debug(f"Expression: {final_expression}")
        logger.debug(f"Implicants: {final_expression.implicants()}")

    def get_safe_sets_brute(self):
        sccs = nx.strongly_connected_components(self.sg)
        sccs = [scc for scc in sccs if len(scc) > 1]

        accepted_sets = []
        for scc in sccs:
            scc_sg = SynthesisGraph()
            scc_g = nx.induced_subgraph(self.sg, scc).copy()
            scc_sg.sg = scc_g

            names = [self.inst(n) for n in scc]
            if all(map(lambda x: x.startswith("PSEUDO-"), names)):
                self.sg.remove_nodes_from(scc)
                continue

            next_round = list(scc)
            current_round = 1

            this_scc_safe = []

            while current_round <= len(next_round):
                pot_safe_sets = [
                    [x] for x in itertools.combinations(next_round, current_round)
                ]
                next_round = set()
                while pot_safe_sets:
                    pot_set = pot_safe_sets.pop(0)
                    can_implement = scc_sg._is_sufficient_safe_set(pot_set)
                    if can_implement:
                        this_scc_safe.append(pot_set)
                    else:
                        next_round.update(pot_set)
                current_round += 1

            accepted_sets.append(this_scc_safe)

        core_set = []
        for node in self.sg.nodes():
            solutions = list(self.iter_solutions(node))
            if not solutions:
                core_set.append(node)

    def _is_sufficient_safe_set(self, safe_set):
        visited = list()
        wl = list(safe_set)

        while wl:
            current = wl.pop(0)
            if current in visited:
                continue

            for sol in self.iter_solutions(current):
                can_implement = True
                for ni, _ in self.iter_solution(current, sol):
                    if ni not in visited:
                        can_implement = False
                        break

                if can_implement or current in safe_set:
                    for user in self.iter_users(current):
                        if user not in wl and user not in visited:
                            wl.append(user)
                    visited.append(current)
                    break

        all_nodes = self.sg.nodes()
        return all(map(lambda n: n in visited, all_nodes))

    def append_result(self, synth_result):
        self.stats[synth_result.state] += 1

        # Add failed spec to graph as well
        spec = synth_result.spec
        specid = self.add_node(spec)

        if not synth_result.is_success():
            return

        parsed_prog = self._parse_program(synth_result.program)

        if self.is_redundant_solution(spec, parsed_prog):
            self.stats["redundant"] += 1
            return

        # Save to programs
        self.programs[spec].append(parsed_prog)

        for idx, line in enumerate(parsed_prog):
            opcode = line["opcode"]
            opcodeid = self.add_node(opcode)

            tag = self._name_to_tag(synth_result.name)
            logger.info(f"{spec} -> {opcode}")

            self.sg.add_edge(
                specid,
                opcodeid,
                tag=tag,
                lineno=idx,
                ops=line["ops"],
                label=f'({tag}, {idx}, [{",".join(line["ops"])}])',
            )

        # self.to_svg()

    def is_redundant_solution(self, spec, parsed_prog):
        is_seen_solution = []
        for idx, prog in enumerate(self.programs[spec]):
            is_seen_solution.append(True)

            if len(prog) != len(parsed_prog):
                is_seen_solution[idx] = False
                continue

            for p, n in zip(prog, parsed_prog):
                if p["opcode"] != n["opcode"]:
                    is_seen_solution[idx] = False
                    break

        if is_seen_solution and any(is_seen_solution):
            return True

        return False

    #################################################################
    ### Helpers
    #################################################################

    def _parse_program(self, program):
        results = []
        lines = program.split("\n")[1:]

        for line in lines:
            if not line:
                continue

            result = dict()
            opcode = line.split(" ", 1)[0].upper()

            operands = []
            if "[" in line:
                operands = line.split("] ")
                if len(operands) > 1:
                    operands = operands[1].split(",")
                else:
                    operands = []
            elif not line.startswith("pnop"):
                operands = line.split(" ", 1)[1].split(",")

            result["opcode"] = opcode
            result["ops"] = operands

            results.append(result)

        return results

    def _name_to_tag(self, name):
        id = name.rsplit("-", 1)[1][:-4]
        return id

    ###############################################################
    ### Add cosmetic information to nodes/edges
    ###############################################################
    def color(self, node, color):
        self.sg.nodes[node]["fillcolor"] = color
        self.sg.nodes[node]["style"] = "filled"

    ################################################################
    ### Utilities to write/read data to/from files
    ################################################################

    def to_file(self, fname):
        nx.write_gpickle(self.sg, fname)

    def save_stats(self, fname):
        with open(fname, "w") as fd:
            fd.write(yaml.dump(self.stats))

        with open(f"{fname[:-5]}-progs.yaml", "w") as fd:
            fd.write(yaml.dump(self.programs))

    def load_partial_graph(self, gname, progs=None):
        logger.info(f"Loading graph: {gname}")
        self.sg = nx.read_gpickle(gname)

        # Populate the other fields
        for node in self.sg.nodes():
            label = self.sg.nodes[node]["label"]
            self.node_map[node] = label

        self.ctr = len(self.sg.nodes()) - 1

        if progs:
            logger.info(f"Loading progs: {progs}")
            # Reconstruct programs from progs
            with open(progs) as fd:
                self.progs = yaml.load(fd)

    def to_svg(self, fname=None):
        from matplotlib import pyplot as plt
        from networkx.drawing.nx_agraph import graphviz_layout, write_dot

        if not fname:
            fname = tf.NamedTemporaryFile(
                dir="synth-results/", prefix="synthg-", suffix=".svg", delete=False
            ).name

        logger.info(f"Writing to: {fname}")

        plt.figure(3, figsize=(12, 9))
        pos = graphviz_layout(self.sg, prog="dot", root=0, args="-Gsize=12,9")
        nx.draw(self.sg, pos, node_color="w", node_shape="s")
        node_labels = nx.get_node_attributes(self.sg, "data")

        for k, v in node_labels.items():
            node_labels[k] = f"{v}"

        nx.draw_networkx_labels(self.sg, pos, labels=node_labels, font_size=10)

        write_dot(self.sg, fname[:-4] + ".dot")

        plt.savefig(fname)
        plt.show()
        plt.close()


def do_push_back(wq, epochs, synth_result):
    epochs[synth_result.spec] += 1

    if epochs[synth_result.spec] < 32:
        wq.push(synth_result.spec)


def generate(results, wq, outfile=None, continue_partial=None):
    from tqdm import tqdm

    epochs = defaultdict(lambda: 0)
    sg = SynthesisGraph()

    if continue_partial:
        graph_path = continue_partial + ".pkl"
        progs = continue_partial + "-progs.yaml"
        sg.load_partial_graph(graph_path, progs)

    try:
        for res in results:
            st = SynthesisTaskResult.from_json(res)
            logger.info(f"Processing: {st}")
            do_push_back(wq, epochs, st)
            sg.append_result(st)
    except KeyboardInterrupt:
        logger.warning("Got keyboard interrupt; saving!")
    finally:
        # sg.save_stats(outfile + ".yaml")
        sg.to_file(outfile + ".pkl")
        logger.success("Generated .pkl file!")
        # sg.to_svg(outfile + ".svg")

    return sg


def stitch_flag_solutions(sg, flag_file):
    jresults = []
    with open(flag_file) as fd:
        results = yaml.load(fd)
    for name, res in results.items():
        jresults.append(res.__dict__)
    results = jresults

    args.o += "-flags"
    flags_sg = generate(results, wq, args.o, "")

    name_to_idx = dict()
    flag_to_idx = dict()

    for node in sg.sg.nodes():
        name = sg.sg.nodes[node]["label"]
        name_to_idx[name] = node

    for node in flags_sg.sg.nodes():
        name = flags_sg.sg.nodes[node]["label"]
        flag_to_idx[name] = node

    # Now, append flags results to the original synthesis graph
    for node in flags_sg.sg.nodes():
        name = flags_sg.sg.nodes[node]["label"]
        flag_solutions = list(flags_sg.iter_solutions(node))

        if name == "TESTQ-R64-R64":
            logger.debug(flag_solutions)

        # No solutions available
        if not flag_solutions:
            continue

        sg_node = name_to_idx[name]

        register_solutions = list(sg.iter_solutions(sg_node))
        additions = []

        for sol in flag_solutions:
            this_add = []
            for ni, op in flags_sg.iter_solution(node, sol):
                nn = flags_sg.sg.nodes[ni]["label"]
                if nn not in name_to_idx:
                    logger.warning(f"Missing {nn}")
                    continue
                reg_node = name_to_idx[nn]
                this_add.append(reg_node)
            additions.append(this_add)

        for sol in register_solutions:
            # Iterate over all the edges of this solution first
            this_solution = list(sg.iter_solution(sg_node, sol))

            for idx, this_flag_solution in enumerate(additions):
                solid = sol
                edge_idx = len(this_solution)
                if idx > 0:
                    # Need to create a new solution id and copy over all the nodes in the
                    # original solution to this new solution before adding in the flag
                    # solutions.
                    solid = str(uuid.uuid4().hex)[:4]
                    for sidx, item in enumerate(this_solution):
                        nn, data = item
                        sg.sg.add_edge(
                            sg_node,
                            nn,
                            tag=solid,
                            lineno=sidx,
                            ops=data,
                            label=f"({solid}, {sidx}, {data})",
                        )

                for nn in this_flag_solution:
                    if name == "TESTQ-R64-R64":
                        logger.debug(
                            f"{solid} {edge_idx} {sg.sg.nodes[sg_node]['label']}"
                        )
                        logger.debug(f"{sg.sg.nodes[nn]['label']}")

                    sg.sg.add_edge(
                        sg_node,
                        nn,
                        tag=solid,
                        lineno=edge_idx,
                        ops="TODO",
                        label="({solid}, {edge_idx}, TODO)",
                    )
                    edge_idx += 1


def append_equivalences(sg, result_files, outfile):
    # Assumption: The log file names are in the same path and named the same as result
    # files, except, ends with a .log instead of a .yaml.
    nodemap = dict()
    for node in sg.sg.nodes():
        name = sg.sg.nodes[node]["label"]
        nodemap[name] = node

    for file in result_files:
        equivalences = dict()
        task_sfx_to_name = dict()
        name = file[:-4] + "log"

        with open(name) as fd:
            logdata = fd.read().split("\n")

        for line in logdata:
            matches = re.search(r"Added task ([0-9a-f]+): (pseudo-.+)", line)
            if matches:
                sfx = matches.group(1)
                name = matches.group(2).upper()
                task_sfx_to_name[sfx] = name
                continue

            matches = re.search(
                r"Found duplicate ([0-9a-f]+) (pseudo-\S+) ([0-9a-f]+)", line
            )
            if matches:
                name = matches.group(2).upper()
                hash = matches.group(3)
                if hash in task_sfx_to_name:
                    eqname = task_sfx_to_name[hash]
                    equivalences[name] = eqname
                    logger.info(f"Added equiv: {name} -> {eqname}")
                else:
                    logger.warning(f"Unknown suffix: {hash}?")

        for name, node in nodemap.items():
            if name in equivalences:
                eqname = equivalences[name]
                eqnode = nodemap[eqname]
                tag = str(uuid.uuid4().hex)[:4]
                if node == eqnode:
                    logger.warning(f"STOPPED ADDING A SELF EDGE: {node}")
                    continue

                sg.sg.add_edge(
                    node, eqnode, tag=tag, lineno=0, ops="", label=f'({tag}, 0, "")'
                )
                logger.info(f"Added edge {name} ({node}) -> {eqname} ({eqnode}) {tag}")

    # Cleanup unused pseudo-instructions with no outgoing edges
    prune_solutions(sg)

    sg.to_file(outfile + ".pkl")
    logger.success("Generated .pkl file!")


def prune_solutions(sg):
    wl = list(sg.sg.nodes())
    removed = list()
    remove_nodes = list()

    while wl:
        to_remove = list()
        node = wl.pop(0)

        solutions = sg.iter_solutions(node)
        label = sg.sg.nodes[node]["label"]

        if solutions:
            continue

        if not label.startswith("PSEUDO"):
            continue

        for user in list(sg.iter_users(node)):
            data = sg.sg.get_edge_data(user, node)
            for _, item in data.items():
                stag = item["tag"]
                if stag not in removed:
                    removed.append(stag)
                    to_remove.append((user, stag))

        if node not in remove_nodes:
            remove_nodes.append(node)

        for user, stag in to_remove:
            sg.discard_solution(user, stag)
            if user not in wl:
                wl.append(user)

    for node in remove_nodes:
        sg.sg.remove_node(node)


def get_sccs(sg):
    sccs = nx.strongly_connected_components(sg)
    return [scc for scc in sccs if len(scc) > 1]


def analyze_scc(sg, scc):
    loop_solutions = set()
    # Get solution(s) that cause the loop
    for node in scc:
        solutions = sg.iter_solutions(node)
        for solution in solutions:
            for ni, _ in sg.iter_solution(node, solution):
                if ni in scc:
                    loop_solutions.add((node, solution))
                    break

    loop_solutions = list(loop_solutions)
    for node, solution in loop_solutions:
        sg.discard_solution(node, solution)

    # Any remaining solutions for the nodes in the SCC must purely be composed of nodes not a part of this SCC
    non_loop_solutions = []
    for node in scc:
        for solution in sg.iter_solutions(node):
            for lineno, val in enumerate(sg.iter_solution(node, solution)):
                ni, operands = val
                non_loop_solutions.append((node, ni, solution, lineno, operands))

    # Create a "merged" node
    merged_name = " v ".join([sg.sg.nodes[nidx]["label"] for nidx in scc])
    merged_name = f"({merged_name})"
    merged_idx = sg.add_node(merged_name)

    # Redirect all users to the merged node
    for node in scc:
        for user in sg.iter_users(node):
            data = sg.sg.get_edge_data(user, node)
            for _, item in data.items():
                sg.sg.add_edge(user, merged_idx, **item)
            sg.sg.remove_edge(user, node)

    # Add edges for all non-loop solutions
    for sol in non_loop_solutions:
        sg.sg.add_edge(merged_idx, sol[1], tag=sol[2], lineno=sol[3], ops=sol[4])

    # Cleanup and remove the nodes in scc
    for node in scc:
        sg.sg.remove_node(node)


def get_leaves(sg):
    leaves = []
    for node in sg.sg.nodes():
        solutions = sg.iter_solutions(node)
        if not solutions:
            leaves.append(node)

    return [sg.sg.nodes[node]["label"] for node in leaves]


def iter_safe_sets(sg):
    leaves = get_leaves(sg)
    yield leaves


def convert_to_dag(sg):
    sccs = get_sccs(sg.sg)
    for scc in sccs:
        analyze_scc(sg, scc)


def get_program_length_distribution(solutions, safe_set, metric=np.min):
    prog_lengths = defaultdict(lambda: 0)
    program_paths = defaultdict(list)
    worklist = []

    for node in list(reversed(list(nx.topological_sort(solutions.sg)))):
        inst = solutions.sg.nodes[node]["label"]
        if inst in safe_set:
            prog_lengths[inst] = 1
            program_paths[inst] = set([inst])
        else:
            worklist.append(node)

    for node in worklist:
        process_node(solutions, node, prog_lengths, program_paths, metric)

    # Filter out pseudo instructions
    for key in list(prog_lengths.keys()):
        if key.startswith("PSEUDO-"):
            del prog_lengths[key]

    return prog_lengths, program_paths


def process_node(solutions, node, prog_lengths, program_paths, metric):
    # No-op, already processed
    cur_inst = solutions.sg.nodes[node]["label"]
    if cur_inst in prog_lengths:
        return

    current_distribution = []
    inst_distributions = []

    for sol in solutions.iter_solutions(node):
        length_distributions = [0]
        inst_distributions.append(set())

        for nn, op in solutions.iter_solution(node, sol):
            process_node(solutions, nn, prog_lengths, program_paths, metric)
            nn_inst = solutions.sg.nodes[nn]["label"]
            length_distributions.append(prog_lengths[nn_inst])

            inst_distributions[-1].update(program_paths[nn_inst])

        length_distributions = np.sum(length_distributions)
        current_distribution.append(length_distributions)

    if current_distribution:
        value = metric(current_distribution)
        prog_lengths[cur_inst] = value

        # Keep track to be able to reconstruct the solution
        solidx = current_distribution.index(value)
        sol = inst_distributions[solidx]
        program_paths[cur_inst] = sol


def analyze_synthesis_graph(sg, outfile):
    convert_to_dag(sg)

    while True:
        try:
            cycle = nx.find_cycle(sg.sg)
            for item in cycle:
                sg.sg.remove_edge(item[0], item[1])
                logger.warning(f"Removed edge: {item[0]}, {item[1]}")
        except:
            break

    safe_set = list(iter_safe_sets(sg))[0]
    safe_set = [x for x in safe_set]

    logger.info(f"safe set: {sorted(safe_set)}")

    name_to_idx = dict()

    for node in sg.sg.nodes():
        name = sg.sg.nodes[node]["label"]
        name_to_idx[name] = node

    length_distribution_min, pp = get_program_length_distribution(
        sg, safe_set, metric=np.min
    )
    length_distribution_max, _ = get_program_length_distribution(
        sg, safe_set, metric=np.max
    )
    table = dict()
    fan_in = defaultdict(lambda: 0)

    for inst, val in sorted(
        length_distribution_min.items(), key=lambda x: (x[1], x[0])
    ):
        if inst in safe_set:
            continue
        table[inst] = {
            "min": val,
            "max": length_distribution_max[inst],
            "inst_used": pp[inst],
        }
        for sinst in pp[inst]:
            fan_in[sinst] += 1

    choices = [x for x in safe_set if " V " in x]

    logger.debug(f"Safe set size: {len(safe_set)}")
    logger.debug(f"Safe set choices: {len(choices)}")

    choices = [len(x.split(" V ")) for x in choices]

    logger.debug(f"# of Insts in choices: {np.sum(choices)}")
    logger.debug(f"Median of Insts in choices: {np.median(choices)}")
    logger.debug(f"Total number of sets: {np.prod(choices)}")

    results = dict()
    results["min_distribution"] = length_distribution_min
    results["max_distribution"] = length_distribution_max
    results["safe_set"] = safe_set
    results["fan_in"] = fan_in
    results["inst_distribution"] = pp

    return results


if __name__ == "__main__":
    argp = argparse.ArgumentParser()

    argp.add_argument("--pipe", type=str, help="Read results from a streaming pipe")

    argp.add_argument("-o", type=str, help="Write output to filename")

    argp.add_argument(
        "--result-file", nargs="+", type=str, help="Read results from result file"
    )

    argp.add_argument(
        "--continue-partial",
        type=str,
        help="Reload and continue synthesis from a previous synthesis graph",
    )

    argp.add_argument(
        "--flags", type=str, help="File containing syntheis results for flags"
    )

    args = argp.parse_args()

    assert args.pipe or args.result_file

    results = []
    wq = ListWQ([])

    if args.pipe:
        with open(args.pipe) as fd:
            conn = yaml.load(fd)

        # XXX
        conn["wq"] = "synth-results"
        results = BeanstalkWQ(conn)

        conn["wq"] = "synth-wq"
        wq = BeanstalkWQ(conn)
    else:
        jresults = []
        for file in args.result_file:
            with open(file) as fd:
                results = yaml.load(fd)
            for name, res in results.items():
                jresults.append(res.__dict__)
        results = jresults

    with open(
        "cluster-results/synth-results-apr-24-2021/inst-parse-stats-ea39.yaml"
    ) as fd:
        inst_parse_stats = yaml.load(fd)
        successes = inst_parse_stats["success"]

    all_inst_count = 0
    for k, v in inst_parse_stats.items():
        all_inst_count += len(v)

    success_names = [
        x.rsplit("/", 1)[1][:-2].upper().replace("_", "-") for x in successes
    ]

    sg = generate(results, wq, args.o, args.continue_partial)
    append_equivalences(sg, args.result_file, args.o)

    sg.get_safe_sets_brute()

    # Print some basic stats
    instructions = []
    have_solutions = []
    for node in sg.sg.nodes():
        name = sg.sg.nodes[node]["label"]
        if name.startswith("PSEUDO"):
            continue
        instructions.append(name)
        solutions = list(sg.iter_solutions(node))
        if solutions:
            have_solutions.append(name)

    logger.debug(f"All X86 Instructions: {all_inst_count}")
    logger.debug(f"Instructions: {len(instructions)}")
    logger.debug(f"Solutions: {len(have_solutions)}")
    logger.debug(f"Solutions: {set(success_names).difference(set(instructions))}")

    print(
        " ".join([f"'{x}'" for x in set(success_names).difference(set(instructions))])
    )

    if args.flags:
        stitch_flag_solutions(sg, args.flags)
        sg.to_file(args.o + ".pkl")
        logger.success(f"Generated {args.o}.pkl file!")

    analyze_synthesis_graph(sg, args.o)
    sg.to_file(args.o + ".pkl")
    logger.success(f"Generated {args.o}.pkl file!")
