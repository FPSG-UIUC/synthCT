; Defines some functions and structs to simulate machine state
#lang rosette/safe

(require rosette/lib/angelic
         rosette/lib/match)

(require rosette/solver/smt/z3)

(provide (all-defined-out))

; Define CPU state and some convinience wrappers to access the define
; registers
(struct state (F Rn) #:mutable #:transparent)

; Some helpers to read/write registers
(define (state-Rn-ref S n)
  (vector-ref (state-Rn S) n))

(define (state-Rn-set! S n v)
  (vector-set! (state-Rn S) n v))

; Some helpers to read/write flags
(define (state-F-ref S n)
  (vector-ref (state-F S) n))

(define (state-F-set! S n v)
  (vector-set! (state-F S) n v))

; An x86-like machine has 16 64 bit registers with subregisters of smaller
; bitwidths also accessible
(define-symbolic R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15 (bitvector 64))

; Only model the important EFLAGS
(define-symbolic CF PF AF ZF SF DF OF (bitvector 1))

; Return a new state
(define (make-state)
  (state (vector CF PF AF ZF SF DF OF)
         (vector R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15)))

; Convinence wrappers to access registers
(define (state-R0 S) (state-Rn-ref S 0))
(define (state-R1 S) (state-Rn-ref S 1))
(define (state-R2 S) (state-Rn-ref S 2))
(define (state-R3 S) (state-Rn-ref S 3))
(define (state-R4 S) (state-Rn-ref S 4))
(define (state-R5 S) (state-Rn-ref S 5))
(define (state-R6 S) (state-Rn-ref S 6))
(define (state-R7 S) (state-Rn-ref S 7))
(define (state-R8 S) (state-Rn-ref S 8))
(define (state-R9 S) (state-Rn-ref S 9))
(define (state-R10 S) (state-Rn-ref S 10))
(define (state-R11 S) (state-Rn-ref S 11))
(define (state-R12 S) (state-Rn-ref S 12))
(define (state-R13 S) (state-Rn-ref S 13))
(define (state-R14 S) (state-Rn-ref S 14))
(define (state-R15 S) (state-Rn-ref S 15))

; Convinence wrappers to access flags
(define (state-CF S) (state-F-ref S 0))
(define (state-PF S) (state-F-ref S 1))
(define (state-AF S) (state-F-ref S 2))
(define (state-ZF S) (state-F-ref S 3))
(define (state-SF S) (state-F-ref S 4))
(define (state-DF S) (state-F-ref S 5))
(define (state-OF S) (state-F-ref S 6))

; Provide a mechanism to undefine
(define (undefine n)
  (define-symbolic* ud (bitvector n)) ud)

(define (undef-bool)
  (define-symbolic* ud boolean?) ud)


; Fixed hardware registers, map them to particular abstract registers
; 4 = rax
; 5 = rcx
; 6 = rdx
(define (read-hw-rax S)
  (state-Rn-ref S 4))
(define (write-hw-rax! S v)
  (state-Rn-set! S 4 v))

(define (read-hw-rcx S)
  (state-Rn-ref S 5))
(define (write-hw-rcx! S v)
  (state-Rn-set! S 5 v))

(define (read-hw-rdx S)
  (state-Rn-ref S 6))
(define (write-hw-rdx! S v)
  (state-Rn-set! S 6 v))
