# Implements base class for factorization


class SubgoalsGenerator:
    def __init__(self, si):
        self.si = si
        self.pseudo = si.pseudo

    # Generic helper functions that need to be accessible to all types of generators to
    # help generate subgoals

    def _is_inst_trivial(self, inst, reg):
        # Sem is trivial if the root of the ast has no outgoing edges
        sem = inst.sems[reg]
        root = sem.root
        if not list(sem.successors(root)):
            return True
        return False


from .bottom_up import FactorizeBottomUp  # noqa: E402
from .top_down import FactorizeTopDown  # noqa: E402


FACTORIZERS = {
    "bottom_up": FactorizeBottomUp,
    "top_down": FactorizeTopDown,
}
