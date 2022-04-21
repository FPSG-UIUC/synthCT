# Implements type inference pass on the AST

import copy

from loguru import logger

# from semantics.containers import ConstOperand

import archinfo

amd64 =  dict([(x.name, x.size) for x in archinfo.ArchAMD64().register_list])

_flags = [
    "CF",
    "PF",
    "AF",
    "ZF",
    "SF",
    "TF",
    "IF",
    "DF",
    "OF",
]


class Typing:
    KEY = "type"

    def __init__(self, sem, normalized=True):
        self.sem = sem
        self.normalized_extracts = normalized
        self.typemap = dict()

    @staticmethod
    def infer(sem, st=None, normalized=True):
        typing = Typing(sem, normalized)
        if st is None:
            st = sem.root
        typing.infer_subtree(st)
        typing.assign_types()

    def infer_subtree(self, idx):
        current = self.sem.node_data(idx, "data")
        succs = list(self.sem.successors_ordered(idx))
        for succ in succs:
            if succ not in self.typemap:
                self.infer_subtree(succ)

        sz = None
        current_type = None

        # Check if this node is an input operand
        for op in self.sem.instsem.operands:
            # if isinstance(op, ConstOperand):
            # continue

            name = op.name
            if ":" in name:
                name, sz = name.split(":")
            if name == current:
                # XXX: This is a hack!
                current_type = BitVec(64)
                break

        if current_type:
            # No-op, we already have the type
            pass
        elif isinstance(current, int):
            current_type = Null()
        elif current == "extractMInt":
            # Output size = high - low
            # Output type = BitVec(output_size)

            high = self.sem.node_data(succs[2], "data")
            low = self.sem.node_data(succs[1], "data")

            # XXX: THIS IS A HACK
            if isinstance(high, str) or isinstance(low, str):
                sz = 1
            elif self.normalized_extracts:
                sz = high - low + 1
            else:
                sz = high - low
            current_type = BitVec(sz)
        elif current == "concatenateMInt":
            # Output size = lhs.sz + rhs.sz
            # Output type = BitVec(output_size)
            tya = self.typemap[succs[0]]
            tyb = self.typemap[succs[1]]
            sz = tya.sz + tyb.sz
            current_type = BitVec(sz)
        elif current == "#ifMInt":
            # Output sz/ty = then.output.sz/ty == else.output.sz/ty
            current_type = copy.copy(self.typemap[succs[1]])
        elif current == "eqMInt":
            # Output sz/ty = 1/bool
            current_type = Bool()
        elif current == "bool2bv":
            current_type = BitVec(1)
        elif current == "bv2bool":
            current_type = Bool()
        elif current in ["ugeMInt", "ugtMInt", "ultMInt", "sgtMInt", "sgeMInt"]:
            current_type = Bool()
        elif current in ["zextMInt", "sextMInt"]:
            # Current type is a BitVec that's indicated by the second argument
            sz = self.sem.node_data(succs[1], "data")
            current_type = BitVec(sz)
        elif current == "mi":
            # Output ty.size = bitvec.lhs.sz
            sz = self.sem.node_data(succs[0], "data")
            current_type = BitVec(sz)
        elif current in _flags:
            # Output ty/sz = bitvec.1
            current_type = BitVec(1)
        elif current.startswith("%") and current[1:] in amd64:
            sz = amd64[current[1:]] * 8
            current_type = BitVec(sz)
            # TODO: The sem needs to keep track of abstract input operands to
            # it.
        elif current == "undefMInt":
            # Assumption: This only appears on RHS of flags
            current_type = BitVec(1)
        elif current in ["undefBool", "false", "true"]:
            current_type = Bool()
        else:
            # Every other operand is simply copy over the type from one of its
            # children/operands.
            if succs:
                current_type = copy.copy(self.typemap[succs[0]])
            else:
                logger.warning(f"Cannot infer type for: {current}; Assigning bv1")
                current_type = BitVec(1)

        self.typemap[idx] = current_type

    def assign_types(self):
        for idx, ty in self.typemap.items():
            self.sem.add_node_data(idx, Typing.KEY, ty)


class Type:
    def __init__(self, sz):
        self.sz = sz

    def __str__(self):
        raise NotImplementedError("Unimplemented!")


class BitVec(Type):
    def __init__(self, sz):
        super().__init__(sz)

    def __str__(self):
        return f"bv{self.sz}"

    def __copy__(self):
        return BitVec(self.sz)


class Bool(Type):
    def __init__(self):
        super().__init__(1)

    def __str__(self):
        return "bool"

    def __copy__(self):
        return Bool()


class Null(Type):
    def __init__(self):
        super().__init__(0)

    def __str__(self):
        return "Null"

    def __copy__(self):
        return Null()
