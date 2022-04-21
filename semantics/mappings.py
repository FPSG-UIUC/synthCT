# Mapping from internal AST opcodes to SMTLIB opcodes
import operator as op
import claripy as cp


def _extract(bv, high, low):
    return op.itemgetter(slice(high, low))(bv)


def _mi(sz, value):
    return cp.BVV(value, sz)


_INFO = {
    "#ifMInt": {"smt": "ite", "claripy": cp.If, "rkt": "kifMInt"},
    "andBool": {"smt": "bvand", "claripy": op.__and__, "rkt": "kandBool"},
    "notBool": {"smt": "bvnot", "claripy": op.__not__, "rkt": "knotBool"},
    "xorBool": {"smt": "bvxor", "claripy": op.__xor__, "rkt": "kxorBool"},
    "orBool": {"smt": "bvor", "claripy": op.__or__, "rkt": "korBool"},
    "==Bool": {"smt": "bveq", "claripy": op.__eq__, "rkt": "keqBool"},
    "=/=Bool": {"smt": "bvne", "claripy": op.__ne__, "rkt": "kneqBool"},
    "impliesBool": {"smt": "=>", "claripy": None},  # TODO
    "andThenBool": {"smt": "", "claripy": None},  # TODO
    "orElseBool": {"smt": "", "claripy": None},  # TODO
    # Define a machine int of (sz, value)
    "mi": {"smt": "", "claripy": _mi, "rkt": "kmi"},
    # Bitwidth manipulation
    "extractMInt": {"smt": "extract", "claripy": _extract, "rkt": "kextractMInt"},
    "concatenateMInt": {
        "smt": "concat",
        "claripy": cp.Concat,
        "rkt": "kconcatenateMInt",
    },
    # Basic-ops
    "addMInt": {"smt": "bvadd", "claripy": op.__add__, "rkt": "kaddMInt"},
    "subMInt": {"smt": "bvsub", "claripy": op.__sub__, "rkt": "ksubMInt"},
    "andMInt": {"smt": "bvand", "claripy": op.__and__, "rkt": "kandMInt"},
    "uremMInt": {"smt": "", "claripy": None, "rkt": "kuremMInt"},
    "xorMInt": {"smt": "", "claripy": op.__xor__, "rkt": "kxorMInt"},
    "orMInt": {"smt": "", "claripy": op.__or__, "rkt": "korMInt"},
    "eqMInt": {"smt": "bveq", "claripy": op.__eq__, "rkt": "keqMInt"},
    "lshrMInt": {"smt": "", "claripy": None, "rkt": "klshrMInt"},
    "rol": {"smt": "", "claripy": None, "rkt": "krolMInt"},
    "ror": {"smt": "", "claripy": None, "rkt": "krorMInt"},
    "mulMInt": {"smt": "", "claripy": None, "rkt": "kmulMInt"},
    "negMInt": {"smt": "", "claripy": None, "rkt": "knegMInt"},
    "shiftLeftMInt": {"smt": "", "claripy": None, "rkt": "kshiftLeftMInt"},
    "aShiftRightMInt": {"smt": "", "claripy": None, "rkt": "kaShiftRightMInt"},
    # Fake K-operations introduced by us in our framework
    "zextMInt": {"smt": "", "claripy": None, "rkt": "kzextMInt"},
    "sextMInt": {"smt": "", "claripy": None, "rkt": "ksextMInt"},
    "bool2bv": {"smt": "", "claripy": None, "rkt": "kbool2bv"},
    "bv2bool": {"smt": "", "claripy": None, "rkt": "kbv2bool"},
    # Signed-ness stuff?
    "svalueMInt": {"smt": "", "claripy": None, "rkt": "ksvalueMInt"},
    "uvalueMInt": {"smt": "", "claripy": None, "rkt": "kuvalueMInt"},
    # Scans
    "scanForward": {"smt": "", "claripy": None},
    "scanReverse": {"smt": "", "claripy": None},
    # No-ops
    "getFlag": {"smt": "", "claripy": None},
    "getParentValue": {"smt": "", "claripy": None},
    "getRegisterValue": {"smt": "", "claripy": None},
    # Misc
    "approx_reciprocal_single": {"smt": "", "claripy": None},
    "approx_reciprocal_sqrt_single": {"smt": "", "claripy": None},
    # Division stuff
    "div_quotient_int16": {"smt": "", "claripy": None, "rkt": "kuDiv"},
    "div_quotient_int32": {"smt": "", "claripy": None, "rkt": "kuDiv"},
    "div_quotient_int64": {"smt": "", "claripy": None, "rkt": "kuDiv"},
    "div_quotient_int8": {"smt": "", "claripy": None, "rkt": "kuDiv"},
    "div_remainder_int16": {"smt": "", "claripy": None, "rkt": "kuRem"},
    "div_remainder_int32": {"smt": "", "claripy": None, "rkt": "kuRem"},
    "div_remainder_int64": {"smt": "", "claripy": None, "rkt": "kuRem"},
    "div_remainder_int8": {"smt": "", "claripy": None, "rkt": "kuRem"},
    "idiv_quotient_int16": {"smt": "", "claripy": None, "rkt": "ksDiv"},
    "idiv_quotient_int32": {"smt": "", "claripy": None, "rkt": "ksDiv"},
    "idiv_quotient_int64": {"smt": "", "claripy": None, "rkt": "ksDiv"},
    "idiv_quotient_int8": {"smt": "", "claripy": None, "rkt": "ksDiv"},
    "idiv_remainder_int16": {"smt": "", "claripy": None, "rkt": "ksRem"},
    "idiv_remainder_int32": {"smt": "", "claripy": None, "rkt": "ksRem"},
    "idiv_remainder_int64": {"smt": "", "claripy": None, "rkt": "ksRem"},
    "idiv_remainder_int8": {"smt": "", "claripy": None, "rkt": "ksRem"},
    # Compares
    "sgtMInt": {"smt": "", "claripy": None, "rkt": "ksgtMInt"},
    "sltMInt": {"smt": "", "claripy": None, "rkt": "ksltMInt"},
    "ugeMInt": {"smt": "", "claripy": None, "rkt": "kugeMInt"},
    "ugtMInt": {"smt": "", "claripy": None, "rkt": "kugtMInt"},
    "ultMInt": {"smt": "", "claripy": None, "rkt": "kultMInt"},
    # Floating-point opcodes
    "add_double": {"smt": "", "claripy": None},
    "add_single": {"smt": "", "claripy": None},
    "div_double": {"smt": "", "claripy": None},
    "div_single": {"smt": "", "claripy": None},
    "maxcmp_double": {"smt": "", "claripy": None},
    "maxcmp_single": {"smt": "", "claripy": None},
    "mincmp_double": {"smt": "", "claripy": None},
    "mincmp_single": {"smt": "", "claripy": None},
    "mul_double": {"smt": "", "claripy": None},
    "mul_single": {"smt": "", "claripy": None},
    "sqrt_double": {"smt": "", "claripy": None},
    "sqrt_single": {"smt": "", "claripy": None},
    "sub_double": {"smt": "", "claripy": None},
    "sub_single": {"smt": "", "claripy": None},
    # Conversion functions
    "comisd": {"smt": "", "claripy": None},
    "comiss": {"smt": "", "claripy": None},
    "cvt_double_to_int32": {"smt": "", "claripy": None},
    "cvt_double_to_int32_truncate": {"smt": "", "claripy": None},
    "cvt_double_to_int64": {"smt": "", "claripy": None},
    "cvt_double_to_int64_truncate": {"smt": "", "claripy": None},
    "cvt_double_to_single": {"smt": "", "claripy": None},
    "cvt_half_to_single": {"smt": "", "claripy": None},
    "cvt_int32_to_double": {"smt": "", "claripy": None},
    "cvt_int32_to_single": {"smt": "", "claripy": None},
    "cvt_int64_to_double": {"smt": "", "claripy": None},
    "cvt_int64_to_single": {"smt": "", "claripy": None},
    "cvt_single_to_double": {"smt": "", "claripy": None},
    "cvt_single_to_int32": {"smt": "", "claripy": None},
    "cvt_single_to_int32_truncate": {"smt": "", "claripy": None},
    "cvt_single_to_int64": {"smt": "", "claripy": None},
    "cvt_single_to_int64_truncate": {"smt": "", "claripy": None},
    # No idea what there are below
    "vfmadd132_double": {"smt": "", "claripy": None},
    "vfmadd132_single": {"smt": "", "claripy": None},
    "vfmadd213_double": {"smt": "", "claripy": None},
    "vfmadd213_single": {"smt": "", "claripy": None},
    "vfmadd231_double": {"smt": "", "claripy": None},
    "vfmadd231_single": {"smt": "", "claripy": None},
    "vfmsub132_double": {"smt": "", "claripy": None},
    "vfmsub132_single": {"smt": "", "claripy": None},
    "vfmsub213_double": {"smt": "", "claripy": None},
    "vfmsub213_single": {"smt": "", "claripy": None},
    "vfmsub231_double": {"smt": "", "claripy": None},
    "vfmsub231_single": {"smt": "", "claripy": None},
    "vfnmadd132_double": {"smt": "", "claripy": None},
    "vfnmadd132_single": {"smt": "", "claripy": None},
    "vfnmadd213_double": {"smt": "", "claripy": None},
    "vfnmadd213_single": {"smt": "", "claripy": None},
    "vfnmadd231_double": {"smt": "", "claripy": None},
    "vfnmsub132_double": {"smt": "", "claripy": None},
    "vfnmsub132_single": {"smt": "", "claripy": None},
    "vfnmsub213_double": {"smt": "", "claripy": None},
}


def to_smt(token):
    return _INFO[token]["smt"]


def to_claripy(token):
    if token in _INFO:
        return _INFO[token]["claripy"]
    return None


def to_rosette(token):
    if token in _INFO:
        if "rkt" in _INFO[token]:
            return _INFO[token]["rkt"]
        print(f"[x] Translation to rosette not found for {token}")
    return None
