# Implements classes used to return results from a synthesis task

from collections import defaultdict


class SynthesisTaskResult:
    def __init__(self, st, state, name, data=None, debug=None, time=-1,
                 is_flag_result=False):
        # XXX: DO NOT MUTATE `st`
        self.name = name
        self.state = state
        self.data = data
        self.debug = debug
        self.time = time

        self.spec = None
        self.components = []
        self.max_timeout = -1

        # Store some flag related metadata for the results
        # If the solution is not for synthesis of flags, then, have the flags been
        # verified to be correct as a side-effect of computation?
        # None = Unchecked, True = checked correct, False = checked incorrect
        self.flags_verified = None
        self.is_flag_result = is_flag_result

        # Extract some information from the synthesis task
        if st:
            self.spec = st.spec.name
            self.components = [comp.name for comp in st.components]
            self.max_timeout = st.timeout
            self.max_prog_len = st.max_prog_len
            self.priority = st.priority

    @staticmethod
    def from_json(json):
        st = SynthesisTaskResult(None, "err", None)

        st.name = json["name"]
        st.state = json["state"]
        st.data = json["data"]
        st.debug = json["debug"]
        st.time = json["time"]
        st.spec = json["spec"]
        st.components = json["components"]
        st.max_timeout = json["max_timeout"]

        return st

    def is_success(self):
        return self.state == "success"

    def is_timeout(self):
        return self.state == "timeout"

    def is_unsat(self):
        return self.state == "unsat"

    def is_eq(self):
        return self.state == "eq"

    @property
    def task(self):
        return self.name

    @property
    def program(self):
        return self.data

    def pretty_print(self):
        from colorama import init, Fore

        init(autoreset=True)

        if self.state == "err":
            print(f"{Fore.RED}[x] {self.name}: Err!")
            print(f"{Fore.RED}{self.debug}")
        elif self.state == "timeout":
            print(f"{Fore.RED}[x] {self.name}: Timeout!")
        elif self.state == "unsat":
            print(f"{Fore.RED}[x] {self.name}: Unsat!")
            print(f"{Fore.RED}{self.debug}")
        elif self.state == "success":
            print(f"{Fore.GREEN}[*] {self.name}: Success!")
            print(f"{Fore.GREEN}{self.data}")
        else:
            print(self.state)

    def __str__(self):
        res = []

        if self.state == "err":
            res.append(f"[x] {self.name}: Err!")
            res.append(f"{self.debug}")
        elif self.state == "timeout":
            res.append(f"[x] {self.name}: Timeout!")
        elif self.state == "unsat":
            res.append(f"[x] {self.name}: Unsat!")
            res.append(f"{self.debug}")
        elif self.state == "success":
            res.append(f"[*] {self.name}: Success!")
            res.append(f"{self.data}")

        return "\n".join(res)


class SynthesisHistory:
    def __init__(self):
        self.history = defaultdict(
            lambda: {"success": 0, "fail": 0, "prog_len": 0, "tries": [0] * 6}
        )

    def update_history(self, result, st):
        name = st.spec.name

        if result.is_success():
            self.history[name]["success"] += 1
        else:
            self.history[name]["fail"] += 1

        prog_len = st.max_prog_len
        self.history[name]["prog_len"] = max(self.history[name]["prog_len"], prog_len)

        self.history[name]["tries"][prog_len] += 1

    def max_prog_len(self, name):
        return self.history[name]["prog_len"]

    def num_tries(self, name, len):
        return self.history[name]["tries"][len]

    def num_successes(self, name):
        return self.history[name]["success"]

    def num_fail(self, name):
        return self.history[name]["fail"]
