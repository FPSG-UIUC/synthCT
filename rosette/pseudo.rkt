; Implements pseudo instructions to help with synthesis
#lang rosette/safe

(require rosette/lib/angelic
         rosette/lib/match)

(require "machine.rkt" "k.rkt")
(provide (all-defined-out))

; ------------- MOV-IMM-R64 ---------------------------

(struct MOVQ-IMM-R64 (i r2) #:transparent)

(define (interpret-MOVQ-IMM-R64 S i r2)
    (state-Rn-set! S r2 i))

(define (print-MOVQ-IMM-R64 i r2)
  (printf "movq 0x~x, R~s~n" (bitvector->natural i) r2))

; ------------- PNOT ---------------------------

(struct PNOT-R64 (r2) #:transparent)

(define (interpret-PNOT-R64 S r2)
  (state-Rn-set! S r2 (bvneg (state-Rn-ref S r2))))

(define (print-PNOT-R64 r2)
  (printf "pnot R~s~n" r2))

; ------------- PMOVQ ---------------------------

(struct PMOVQ-R64-R64 (r1 r2) #:transparent)

(define (interpret-PMOVQ-R64-R64 S r1 r2)
    (state-Rn-set! S r2 (state-Rn-ref S r1)))

(define (print-PMOVQ-R64-R64 r1 r2)
  (printf "movq R~s, R~s~n" r1 r2))

; ------------- PSPLIT ---------------------------
;
; Brief description of semantics:
; r1 = r2[0:r3]
; r2 = r2[r3:64]

(struct PSPLIT-R64-R64 (r1 r2 r3) #:transparent)

(define (print-PSPLIT-R64-R64 r1 r2 r3)
  (printf "psplit R~s, R~s, R~s~n" r1 r2 r3))

(define (interpret-PSPLIT-R64-R64 S r1 r2 r3)
(let (
      [local-r2 (state-Rn-ref S r2)]
[local-r3 (interpret-k (kandMInt (bv 63 64) (state-Rn-ref S r3)))]
)
  (begin
    (state-Rn-set! S r1
      (interpret-k
        (klshrMInt local-r2 (ksubMInt (bv 64 64) local-r3)))
        )
    (state-Rn-set! S r2
      (interpret-k
        (kandMInt local-r2
            (ksubMInt (kshiftLeftMInt (bv 1 64) (ksubMInt (bv 64 64)
                                                          local-r3))
                      (bv 1 64)))
        ))
    )
  )
)

; ------------- PCONCAT ---------------------------
;
; Brief description of semantics:
; r2 = (r1 << r3) ~ r2

(struct PCONCAT-R32-R32 (r1 r2 r3) #:transparent)

(define (print-PCONCAT-R32-R32 r1 r2 r3)
  (printf "pconcat-r32-r32 R~s, R~s, R~s~n" r1 r2 r3))

(define (interpret-PCONCAT-R32-R32 S r1 r2 r3)
(let (
[local-r1 (state-Rn-ref S r1)]
[local-r2 (state-Rn-ref S r2)]
[local-r3 (interpret-k (kandMInt (bv 63 64) (state-Rn-ref S r3)))]
)
  (begin
    (state-Rn-set! S r2
      (interpret-k
        (korMInt local-r2 (kshiftLeftMInt local-r1 local-r3))))
    )
  )
)


; ---------------------- PMOV-FLAG-R64 ----------------------------
;

(struct PMOV-FLAG-R64 (i r2) #:transparent)

(define (print-PMOV-FLAG-R64 i r2)
  (printf "pmov-flag-r64 F~s, R~s~n" i r2))

(define (interpret-PMOV-FLAG-R64 S i r2)
  (begin
    (assume (and (bvuge i (bv 0 64)) (bvult i (bv 7 64))))
    (state-Rn-set! S r2 (zero-extend (state-F-ref S (bitvector->integer i))
                                     (bitvector 64)))
  )
)


(struct PMOV-R64-FLAG (r1 i) #:transparent)

(define (print-PMOV-R64-FLAG r1 i)
  (printf "pmov-r64-flag R~s, F~s~n" r1 i))

(define (interpret-PMOV-R64-FLAG S r1 i)
  (begin
    (assume (and (bvuge i (bv 0 64)) (bvult i (bv 7 64))))
    (state-F-set! S (bitvector->integer i)
                    (bool->bitvector (bitvector->bool (state-Rn-ref S r1)) 1))
  )
)

(struct PSET-FLAG (i) #:transparent)

(define (print-PSET-FLAG i)
  (printf "pset-flag F~s~n" i))

(define (interpret-PSET-FLAG S i)
  (begin
    (assume (and (bvuge i (bv 0 64)) (bvult i (bv 7 64))))
    (state-F-set! S (bitvector->integer i) (bv 1 1))))

(struct PRESET-FLAG (i) #:transparent)

(define (print-PRESET-FLAG i)
  (printf "preset-flag F~s~n" i))

(define (interpret-PRESET-FLAG S i)
  (begin
    (assume (and (bvuge i (bv 0 64)) (bvult i (bv 7 64))))
    (state-F-set! S (bitvector->integer i) (bv 0 1))))


; ----------------- PINSERT-BIT-R64-R64
;
; Insert a bit, represented as a 64 bitvec, into R2 at location indicated in
; R3. Bits are 1-indexed. Indexing at 0 returns the same BV,
;
; R2 = R2[0:R3] ~ BIT ~ R2[R3:64]


(struct PINSERT-BIT-R64-R64 (r1 r2 r3) #:transparent)

(define (print-PINSERT-BIT-R64-R64 r1 r2 r3)
  (printf "pinsert-bit-r64-r64 R~s, R~s, R~s~n" r1 r2 r3))

(define (interpret-PINSERT-BIT-R64-R64 S r1 r2 r3)
  (let (
        [local-r1  (state-Rn-ref S r1)]
        [local-r2  (state-Rn-ref S r2)]
        [local-r3  (state-Rn-ref S r3)]
        [top-half (bvlshr (state-Rn-ref S r2) (state-Rn-ref S r3))]
        [bottom-half (bvand (state-Rn-ref S r2)
                            (bvsub
                            (bvshl (bv 1 64)
                                   (state-Rn-ref S r3))
                            (bv 1 64)))]
        )
    (begin
      ; Location indicated in local-r3 should be < 64. Add the constraint.
      (assume (and (bvuge local-r3 (bv 0 64)) (bvult local-r3 (bv 64 64))))
      (state-Rn-set! S r2 (bvor (bvshl top-half local-r3)
                              (bvshl local-r1 (bvsub local-r3 (bv 1 64)))
                              (bvlshr bottom-half (bv 1 64)))
                     )
      )
))


; ----- A very simple cmov-like instruction

(struct PCMOV-R64-R64-R64 (r1 r2 r3) #:transparent)

(define (print-PCMOV-R64-R64-R64 r1 r2 r3)
  (printf "pcmov-r64-r64-r64 R~s, R~s, R~s~n" r1 r2 r3))

(define (interpret-PCMOV-R64-R64-R64 S r1 r2 r3)
  (let (
        [local-r1  (state-Rn-ref S r1)]
        [local-r2  (state-Rn-ref S r2)]
        [local-r3  (state-Rn-ref S r3)])
    (begin
      (state-Rn-set! S r2 (if (bitvector->bool local-r1) local-r2 local-r3))
      )
))


(struct PNOP () #:transparent)

(define (print-PNOP)
  (printf "pnop~n"))

(define (interpret-PNOP S)
  #f
  )


(struct PEXTRACT-R64-BIT (i r1) #:transparent)

(define (print-PEXTRACT-R64-BIT i r1)
  (printf "pextract-bit 0x~x, R~s~n" i r1))

(define (interpret-PEXTRACT-R64-BIT S i r1)
  (let (
        [local-r1 (state-Rn-ref S r1)])
  (begin
    (assume (bvule i (bv 63 64)))
    (state-Rn-set! S r1 (extract r1 i i)))
  )
)

(struct POR-R64-R64 (r1 r2) #:transparent)

(define (print-POR-R64-R64 r1 r2)
  (printf "por-r64-r64 R~s, R~s~n" r1 r2))

(define (interpret-POR-R64-R64 S r1 r2)
  (let (
    [local-r1 (state-Rn-ref S r1)]
    [local-r2 (state-Rn-ref S r2)]
    )
  (begin
    (state-Rn-set! S r2 (bvor local-r1 local-r2)))
  )
)


(struct PAND-R64-R64 (r1 r2) #:transparent)

(define (print-PAND-R64-R64 r1 r2)
  (printf "pand-r64-r64 R~s, R~s~n" r1 r2))

(define (interpret-PAND-R64-R64 S r1 r2)
  (let (
    [local-r1 (state-Rn-ref S r1)]
    [local-r2 (state-Rn-ref S r2)]
    )
  (begin
    (state-Rn-set! S r2 (bvand local-r1 local-r2)))
  )
)

(struct PXOR-R64-R64 (r1 r2) #:transparent)

(define (print-PXOR-R64-R64 r1 r2)
  (printf "pxor-r64-r64 R~s, R~s~n" r1 r2))

(define (interpret-PXOR-R64-R64 S r1 r2)
  (let (
    [local-r1 (state-Rn-ref S r1)]
    [local-r2 (state-Rn-ref S r2)]
    )
  (begin
    (state-Rn-set! S r2 (bvxor local-r1 local-r2)))
  )
)
