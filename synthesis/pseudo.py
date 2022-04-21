# Implements classes to read pseudo-instruction spec file and store them

import yaml
from semantics.containers import Operand
from loguru import logger


def load_pseudos_from_yaml(yamlf):
    with open(yamlf) as fd:
        data = yaml.safe_load(fd)

    insts = []
    for key, item in data.items():
        insts.append(PseudoInstruction(key, item))

    return insts


class PseudoInstruction:
    def __init__(self, inst, inst_dict):
        self.inst = inst
        self.props = inst_dict

    def is_pseudo(self):
        return True

    @property
    def name(self):
        return self.inst

    @property
    def struct_name(self):
        return self.inst

    @property
    def defined_in(self):
        return self.props["file"]

    @property
    def operands(self):
        return [Operand(x, 64, "R64") for x in self.props["operands"]]

    @property
    def print_fn(self):
        return f"print-{self.inst}"

    def iter_output_regs(self):
        return [(x, None) for x in self.props["outputs"]]

    @property
    def interpret_fn(self):
        return f"interpret-{self.inst}"
