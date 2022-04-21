#lang rosette/safe

(require rosette/lib/angelic
         rosette/lib/match)

(provide (all-defined-out))

;-------------------------------------------------------
; Define the basic constructs within k-semantics

; Control-flow instruction
(struct kifMInt (c t f) #:transparent)

; Boolean operations
(struct kandBool (a b) #:transparent)
(struct knotBool (a) #:transparent)
(struct kxorBool (a b) #:transparent)
(struct korBool (a b) #:transparent)

; Boolean comparators
(struct keqBool (a b) #:transparent)
(struct kneBool (a b) #:transparent)

(struct ksubMInt (a b) #:transparent)
(struct kandMInt (a b) #:transparent)
(struct kuremMInt (a b) #:transparent)
(struct kxorMInt (a b) #:transparent)
(struct korMInt (a b) #:transparent)

(struct klshrMInt (a b) #:transparent)
(struct krolMInt (a b) #:transparent)
(struct krorMInt (a b) #:transparent)
(struct kmulMInt (a b) #:transparent)
(struct knegMInt (a) #:transparent)
(struct kshiftLeftMInt (a b) #:transparent)
(struct kaShiftRightMInt (a b) #:transparent)

(struct kscanForwardMInt (a b) #:transparent)
(struct kscanReverseMInt (a b) #:transparent)

(struct ksgtMInt (a b) #:transparent)
(struct ksltMInt (a b) #:transparent)
(struct kugeMInt (a b) #:transparent)
(struct kugtMInt (a b) #:transparent)
(struct kultMInt (a b) #:transparent)

; Basic arithmetic
(struct kaddMInt (a b) #:transparent)

; Division (u is for unsigned, s for signed)
(struct kuDiv (a b) #:transparent)
(struct kuRem (a b) #:transparent)
(struct ksDiv (a b) #:transparent)
(struct ksRem (a b) #:transparent)

; Length change on bitvectors
(struct kextractMInt (bv b e) #:transparent)
(struct kconcatenateMInt (a b) #:transparent)

(struct keqMInt (a b) #:transparent)

; Literals
(struct kmi (w v) #:transparent)

; type casting
(struct ksvalueMInt (v) #:transparent)
(struct kuvalueMInt (v) #:transparent)

(struct kzextMInt (x t) #:transparent)
(struct ksextMInt (x t) #:transparent)

(struct kbv2bool (v) #:transparent)
(struct kbool2bv (v) #:transparent)


; Define an interpretter over k-semantics
(define (interpret-k k)
  (match k
    ; Control-flow
    [(kifMInt c t f) (if (interpret-k c) (interpret-k t) (interpret-k f))]
    [(kaddMInt a b) (bvadd (interpret-k a) (interpret-k b))]
    [(kextractMInt bv b e) (extract (interpret-k e) (interpret-k b) (interpret-k bv))]
    [(kconcatenateMInt a b) (concat (interpret-k a) (interpret-k b))]
    [(keqMInt a b) (bveq (interpret-k a) (interpret-k b))]
    [(kmi w v) (bv (interpret-k v) (interpret-k w))]

    [(kandBool a b) (&& (interpret-k a) (interpret-k b))]
    [(knotBool a) (! (interpret-k a))]
    [(kxorBool a b) (xor (interpret-k a) (interpret-k b))]
    [(korBool a b) (|| (interpret-k a) (interpret-k b))]

    [(keqBool a b) (eq? (interpret-k a) (interpret-k b))]
    [(kneBool a b) (not (eq? (interpret-k a) (interpret-k b)))]

    [(ksubMInt a b) (bvsub (interpret-k a) (interpret-k b))]
    [(kandMInt a b) (bvand (interpret-k a) (interpret-k b))]
    [(kuremMInt a b) (bvurem (interpret-k a) (interpret-k b))]
    [(kxorMInt a b) (bvxor (interpret-k a) (interpret-k b))]
    [(korMInt a b) (bvor (interpret-k a) (interpret-k b))]

    [(klshrMInt a b) (bvlshr (interpret-k a) (interpret-k b))]
    [(krolMInt a b) (bvrol (interpret-k a) (interpret-k b))]
    [(krorMInt a b) (bvror (interpret-k a) (interpret-k b))]
    [(kmulMInt a b) (bvmul (interpret-k a) (interpret-k b))]
    [(knegMInt a) (bvneg (interpret-k a))]
    [(kshiftLeftMInt a b) (bvshl (interpret-k a) (interpret-k b))]
    [(kaShiftRightMInt a b) (bvashr (interpret-k a) (interpret-k b))]

    [(ksgtMInt a b) (bvsgt (interpret-k a) (interpret-k b))]
    [(ksltMInt a b) (bvslt (interpret-k a) (interpret-k b))]
    [(kugeMInt a b) (bvuge (interpret-k a) (interpret-k b))]
    [(kugtMInt a b) (bvugt (interpret-k a) (interpret-k b))]
    [(kultMInt a b) (bvult (interpret-k a) (interpret-k b))]

    [(kuvalueMInt v) (bitvector->natural (interpret-k v))]
    [(ksvalueMInt v) (bitvector->integer (interpret-k v))]

    [(ksextMInt x t) (sign-extend (interpret-k x) (bitvector t))]
    [(kzextMInt x t) (zero-extend (interpret-k x) (bitvector t))]

    [(kbv2bool x) (bitvector->bool (interpret-k x))]
    [(kbool2bv x) (bool->bitvector (interpret-k x))]

    [(kuDiv a b) (bvudiv (interpret-k a) (interpret-k b))]
    [(kuRem a b) (bvurem (interpret-k a) (interpret-k b))]
    [(ksDiv a b) (bvsdiv (interpret-k a) (interpret-k b))]
    [(ksRem a b) (bvsrem (interpret-k a) (interpret-k b))]
    
    [v v]
))
