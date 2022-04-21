#lang rosette/safe

(require (only-in racket hash in-range for for/list with-handlers flush-output
                  thread thread-wait break-thread exn:break?
                  make-semaphore semaphore-wait semaphore-post
                  call-with-semaphore/enable-break
                  processor-count))
(require rosette/lib/angelic
         rosette/lib/match)

(require rosette/solver/smt/z3)

; Import from local
(require "machine.rkt")

(provide (all-defined-out))

; Define the sketches
; TODO: The current instruction sketch assumes that an instruction can only
; have at most two operands.
; May need to relax these constraints later on.
; TODO: The current machine has only 64b registers
(define (??insn options)
  (define r1 (choose* 0 1 2 3 4 5 6 7))
  (define r2 (choose* 0 1 2 3 4 5 6 7))
  (define r3 (choose* 0 1 2 3 4 5 6 7))
  (define r4 (choose* 0 1 2 3 4 5 6 7))
  (define r5 (choose* 0 1 2 3 4 5 6 7))
  (define-symbolic* i (bitvector 64))
  (apply choose* (options r1 r2 r3 r4 r5 i)))

(define (??prog fuel options)
  (if (= fuel 0) null
    (cons (??insn options) (??prog (- fuel 1) options))))

(define (??partial-prog prog fuel options)
  (if (= fuel 0) null
    (append (cons (??insn options) (??prog (- fuel 1) options)) prog (cons (??insn options) (??prog (- fuel 1) options)))))

; Program synthesizer
; Takes a sketch to synthesize from and asserts to check for equivalence to
; the original specification
(define (synthesize-prog sketch asserts interpretter)
  (define S (make-state))
  (define S* (make-state))
  (define solution
    (synthesize
      #:forall S
      #:guarantee
      (begin
        (assume (bveq (state-Rn-ref S 1) (state-Rn-ref S* 1)))
        (for-each (lambda (insn) (interpretter S* insn)) sketch)
        (asserts S S*))))
  (if (unsat? solution) #f
    (evaluate sketch solution)))

; Wrapper around synthesize-prog, uses sketch-gen to generate sketch for
; synthesis and explores programs upto length max-fuel incrementing fuel with
; each unsuccessful call to program synthesizer
(define (optimize-prog max-fuel options sketch-gen asserts printer)
  (define (worker fuel)
    (define prog (synthesize-prog (sketch-gen fuel options) asserts))
    (if prog
      (begin
        (eprintf "sat! ~s~n" fuel)
        (displayln prog)
        (for-each printer prog))
      (begin
        (eprintf "unsat! ~s~n" fuel)
        (if (>= fuel max-fuel) #f
          (worker (+ fuel 1))))))
  (worker 0))


(define (optimize-prog-single max-fuel options sketch-gen asserts interpretter printer)
  (define (worker fuel)
    (define prog (synthesize-prog (sketch-gen fuel options) asserts interpretter))
    (if prog
      (begin
        (eprintf "sat! ~s~n" fuel)
        (displayln prog)
        (for-each printer prog))
      (begin
        (eprintf "unsat! ~s~n" fuel))))
  (worker max-fuel))

; Parallel version of above
(define (optimize-prog/parallel max-fuel options sketch-gen asserts printer)
  (define solved (box #f))
  (define solved-fuel (box 1000))
  (define threads (box '()))
  (define report-sema (make-semaphore 1))
  (define (worker fuel)
    (cond
      [(or (not (unbox solved)) (< fuel (unbox solved-fuel)))
       (define prog (synthesize-prog (sketch-gen fuel options) asserts))
       (call-with-semaphore/enable-break report-sema
                                         (lambda ()
                                           (if prog
                                             (begin
                                               (eprintf "sat! ~s~n" fuel)
                                               (for-each (lambda (thd-fuel)
                                                           (if (> (cdr thd-fuel) fuel)
                                                             (break-thread (car thd-fuel))
                                                             (void))) (unbox threads))
                                               (if (or (not (unbox solved)) (< fuel (unbox solved-fuel)))
                                                 (begin
                                                   (set-box! solved-fuel fuel)
                                                   (set-box! solved prog))
                                                 (void)))
                                             (eprintf "unsat! ~s~n" fuel))))]))
  (define core-sema (make-semaphore (processor-count)))
  (for ([fuel (in-range (add1 max-fuel))])
    (semaphore-wait core-sema)
    (define thd
      (thread (lambda ()
                (with-handlers ([exn:break? (lambda (x) (void))])
                               (worker fuel))
                (semaphore-post core-sema))))
    (set-box! threads (cons (cons thd fuel) (unbox threads))))
  (for-each (lambda (thd-fuel) (thread-wait (car thd-fuel))) (unbox threads))
  (if (not (unbox solved)) (void)
    (begin
      (for-each printer (unbox solved))
      (flush-output))))

;----------------------------------------------------
; Top-level function to invoke a Synthesis Task (ST)
; :spec: Specification to synthesize
; :r1 r2: Registers to use; this is usually 0 1
; :options: Subcomponents the synthesis task is allowed to choose from
; :max-fuel: Maximum length of programs to explore
;
; Deprecated
#|(define (synth-insn spec rs options max-fuel asserts)|#
  #|(optimize-prog/parallel|#
    #|max-fuel|#
    #|options|#
    #|(lambda (fuel options) (??prog fuel options))|#
    #|(apply asserts spec rs)|#
#|))|#

(define (synth-insn-single spec rs options max-fuel asserts interpretter printer)
  (optimize-prog-single
    max-fuel
    options
    (lambda (fuel options) (??prog fuel options))
    (apply asserts spec rs)
    interpretter
    printer
))


(define (synth-insn-flag spec rs options max-fuel prog asserts interpretter printer)
  (optimize-prog-single
    max-fuel
    options
    (lambda (fuel options) (??partial-prog prog fuel options))
    (apply asserts spec rs)
    interpretter
    printer
))


;---------------------------------------------------------------
; Debug helper for verifying known solutions


(define (verify-solution prog asserts interpretter)
  (define S (make-state))
  (define S* (make-state))
  (define solution
    (verify
      (begin
            (for-each (lambda (insn) (interpretter S* insn)) prog)
            (asserts S S*))))
  (if (unsat? solution) #t
    (begin
      (displayln solution)
      (evaluate S solution)
      (evaluate S* solution)
      (printf "Ref. R0:  ~s~n" (state-Rn-ref S 0))
      (printf "Impl. R0: ~s~n" (state-Rn-ref S* 0))
      (printf "Ref. R1:  ~s~n" (state-Rn-ref S 1))
      (printf "Impl. R1: ~s~n" (state-Rn-ref S* 1))
    )))


(define (print-solution prog interpretter)
  (define S* (make-state))
  (begin
    (for-each (lambda (insn) (interpretter S* insn)) prog)
        (displayln "===")
        (printf "R0: ~s~n" (state-Rn-ref S* 0))
        (printf "R1: ~s~n" (state-Rn-ref S* 1))
        (printf "R2: ~s~n" (state-Rn-ref S* 2))
        (printf "R3: ~s~n" (state-Rn-ref S* 3))
        (printf "R4: ~s~n" (state-Rn-ref S* 4))
        (printf "R5: ~s~n" (state-Rn-ref S* 5))
        (displayln "===")
        ))

#|(define (synthesize-prog sketch asserts)|#
  #|(define S (make-state))|#
  #|(define S* (make-state))|#
  #|(define solution|#
    #|(synthesize|#
      #|#:forall S|#
      #|#:guarantee|#
      #|(begin|#
        #|(for-each (lambda (insn) (interpret-x86insn S* insn)) sketch)|#
        #|(asserts S S*))))|#
  #|(if (unsat? solution) #f|#
    #|(evaluate sketch solution)))|#
