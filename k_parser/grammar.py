# from parsimonious.grammar import Grammar

# kgrammar = Grammar(
# r"""
# prog = (line)+
# line = (comment / require / module_def / emptyline)

# modline = (comment / imports / rule_def / emptyline)

# comment = ~"//.*"
# require = "requires \"x86-configuration.k\""
# word = ~r"[-\w]+"

# module_def = module_start (modline)* module_end
# module_start = "module " word
# module_end = "endmodule"

# rule_def = "rule <k>" nl ws? exec nl ws? "...</k>"
# exec = "execinstr" ws args ws "=> ."
# args = ~r"\(([^,]+,?)+\)"


# imports = "imports X86-CONFIGURATION"
# ws          = ~"\s*"
# nl          = ~"\n*"
# emptyline   = ws+
# """
# )


cf_opcodes = ["#ifMInt", "#then", "#else", "#fi"]

bool_opcodes = [
    "andBool",
    "notBool",
    "andThenBool",
    "xorBool",
    "orBool",
    "orElseBool",
    "impliesBool",
    "==Bool",
    "=/=Bool",
]

functions = [
    # Bitwidth manipulation
    "extractMInt",
    "concatenateMInt",
    # Basic-ops
    "addMInt",
    "subMInt",
    "andMInt",
    "uremMInt",
    "xorMInt",
    "orMInt",
    "eqMInt",
    "lshrMInt",
    "rol",
    "ror",
    "mulMInt",
    "negMInt",
    "shiftLeftMInt",
    "aShiftRightMInt",
    # Signed-ness stuff?
    "svalueMInt",
    "uvalueMInt",
    # Scans
    "scanForward",
    "scanReverse",
    # No-ops
    "getFlag",
    "getParentValue",
    "getRegisterValue",
    # Define a machine int of (sz, value)
    "mi",
    # Misc
    "approx_reciprocal_single",
    "approx_reciprocal_sqrt_single",
    # Division stuff
    "div_quotient_int16",
    "div_quotient_int32",
    "div_quotient_int64",
    "div_quotient_int8",
    "div_remainder_int16",
    "div_remainder_int32",
    "div_remainder_int64",
    "div_remainder_int8",
    "idiv_quotient_int16",
    "idiv_quotient_int32",
    "idiv_quotient_int64",
    "idiv_quotient_int8",
    "idiv_remainder_int16",
    "idiv_remainder_int32",
    "idiv_remainder_int64",
    "idiv_remainder_int8",
    # Compares
    "sgtMInt",
    "sltMInt",
    "ugeMInt",
    "ugtMInt",
    "ultMInt",
    # Undefine
    "undefMInt",
]

floating_point_functions = [
    "add_double",
    "add_single",
    "div_double",
    "div_single",
    "maxcmp_double",
    "maxcmp_single",
    "mincmp_double",
    "mincmp_single",
    "mul_double",
    "mul_single",
    "sqrt_double",
    "sqrt_single",
    "sub_double",
    "sub_single",
    # Conversion functions
    "comisd",
    "comiss",
    "cvt_double_to_int32",
    "cvt_double_to_int32_truncate",
    "cvt_double_to_int64",
    "cvt_double_to_int64_truncate",
    "cvt_double_to_single",
    "cvt_half_to_single",
    "cvt_int32_to_double",
    "cvt_int32_to_single",
    "cvt_int64_to_double",
    "cvt_int64_to_single",
    "cvt_single_to_double",
    "cvt_single_to_int32",
    "cvt_single_to_int32_truncate",
    "cvt_single_to_int64",
    "cvt_single_to_int64_truncate",
    # No idea what there are below
    "vfmadd132_double",
    "vfmadd132_single",
    "vfmadd213_double",
    "vfmadd213_single",
    "vfmadd231_double",
    "vfmadd231_single",
    "vfmsub132_double",
    "vfmsub132_single",
    "vfmsub213_double",
    "vfmsub213_single",
    "vfmsub231_double",
    "vfmsub231_single",
    "vfnmadd132_double",
    "vfnmadd132_single",
    "vfnmadd213_double",
    "vfnmadd213_single",
    "vfnmadd231_double",
    "vfnmsub132_double",
    "vfnmsub132_single",
    "vfnmsub213_double",
]

all_functions = functions + floating_point_functions
