# Dummy python module to hold a bunch of string constants required to emit
# racket/rosette code.

IMPORTS = """#lang rosette/safe

(require (only-in racket hash in-range for for/list with-handlers flush-output
                  thread thread-wait break-thread exn:break?
                  make-semaphore semaphore-wait semaphore-post call-with-semaphore/enable-break
                  processor-count))
(require rosette/lib/angelic
         rosette/lib/match)

(require rosette/solver/smt/z3)
"""
