from typing import Union


def base_as_int(c: str) -> int:
    """Map one character from bases ACGU to an int 0..3"""
    return "ACGU".find(c)


def int_as_base(r: int) -> str:
    """Map int bases 0..3 to a character from ACGU"""
    return "ACGU"[r]


# Six base pairs may occur in helices.
# Here we can look up the str form from an index we call a "pair-int".
_PAIRS = (
    "AU",
    "CG",
    "GC",
    "GU",
    "UA",
    "UG",
)
# We sometimes want a pair of bases (as ints) from a pair-int.
_INT_PAIRS = tuple((base_as_int(p[0]), base_as_int(p[1])) for i, p in enumerate(_PAIRS))
# Sometimes we'd like to go from the str or tuple form to the pair-int.
_PAIRS_INV: dict[Union[str, tuple[int, int]], int] = {p: i for i, p in enumerate(_PAIRS)}
_PAIRS_INV.update({t: i for i, t in enumerate(_INT_PAIRS)})


def pair_int_as_str(pair_int: int) -> str:
    """Convert pair-int to str form (e.g. 3 -> 'GU')"""
    return _PAIRS[pair_int]


def pair_int_as_tuple(pair_int: int) -> tuple[int, int]:
    return _INT_PAIRS[pair_int]


def str_as_pair_int(p: str) -> int:
    """Convert a pair-str to an index to various tables (or -1)

    p is a string pair like "GU" and the return is an integer index
    in a list of Watson-Crick-Franklin pairs for RNA including GU.
    If p is not one of those pairs, -1 is returned.
    """
    return _PAIRS_INV.get(p, -1)


def tuple_as_pair_int(r: int, s: int) -> int:
    """Convert a pair of bases to an index to various tables (or -1)

    (r,s) is a base pair like (2,3) (signifying "GU") and the return
    is an integer index in a list of Watson-Crick-Franklin pairs for
    RNA including GU. If (r,s) is not one of those pairs, -1 is
    returned.
    """
    return _PAIRS_INV.get((r, s), -1)


# Free energies (-ve is favourable, +ve is a "penalty")
SIMPLE_PAIRING_FE = dict(
    base_pair_AU=-1.6, base_pair_CG=-3.0, base_pair_GU=-0.4, hairpin_01=1
)


# Free energy table based on Turner 2004
TURNER_FE = dict(
    helix_terminal_AUAA=-0.8,
    helix_terminal_AUAC=-1.0,
    helix_terminal_AUAG=-0.8,
    helix_terminal_AUAU=-1.0,
    helix_terminal_AUCA=-0.6,
    helix_terminal_AUCC=-0.7,
    helix_terminal_AUCG=-0.6,
    helix_terminal_AUCU=-0.7,
    helix_terminal_AUGA=-0.8,
    helix_terminal_AUGC=-1.0,
    helix_terminal_AUGG=-0.8,
    helix_terminal_AUGU=-1.0,
    helix_terminal_AUUA=-0.6,
    helix_terminal_AUUC=-0.8,
    helix_terminal_AUUG=-0.6,
    helix_terminal_AUUU=-0.8,
    helix_terminal_CGAA=-1.5,
    helix_terminal_CGAC=-1.5,
    helix_terminal_CGAG=-1.4,
    helix_terminal_CGAU=-1.5,
    helix_terminal_CGCA=-1.0,
    helix_terminal_CGCC=-1.1,
    helix_terminal_CGCG=-1.0,
    helix_terminal_CGCU=-0.8,
    helix_terminal_CGGA=-1.4,
    helix_terminal_CGGC=-1.5,
    helix_terminal_CGGG=-1.6,
    helix_terminal_CGGU=-1.5,
    helix_terminal_CGUA=-1.0,
    helix_terminal_CGUC=-1.4,
    helix_terminal_CGUG=-1.0,
    helix_terminal_CGUU=-1.2,
    helix_terminal_GCAA=-1.1,
    helix_terminal_GCAC=-1.5,
    helix_terminal_GCAG=-1.3,
    helix_terminal_GCAU=-1.5,
    helix_terminal_GCCA=-1.1,
    helix_terminal_GCCC=-0.7,
    helix_terminal_GCCG=-1.1,
    helix_terminal_GCCU=-0.5,
    helix_terminal_GCGA=-1.6,
    helix_terminal_GCGC=-1.5,
    helix_terminal_GCGG=-1.4,
    helix_terminal_GCGU=-1.5,
    helix_terminal_GCUA=-1.1,
    helix_terminal_GCUC=-1.0,
    helix_terminal_GCUG=-1.1,
    helix_terminal_GCUU=-0.7,
    helix_terminal_GUAA=-0.3,
    helix_terminal_GUAC=-1.0,
    helix_terminal_GUAG=-0.8,
    helix_terminal_GUAU=-1.0,
    helix_terminal_GUCA=-0.6,
    helix_terminal_GUCC=-0.7,
    helix_terminal_GUCG=-0.6,
    helix_terminal_GUCU=-0.7,
    helix_terminal_GUGA=-0.6,
    helix_terminal_GUGC=-1.0,
    helix_terminal_GUGG=-0.8,
    helix_terminal_GUGU=-1.0,
    helix_terminal_GUUA=-0.6,
    helix_terminal_GUUC=-0.8,
    helix_terminal_GUUG=-0.6,
    helix_terminal_GUUU=-0.6,
    helix_terminal_UAAA=-1.0,
    helix_terminal_UAAC=-0.8,
    helix_terminal_UAAG=-1.1,
    helix_terminal_UAAU=-0.8,
    helix_terminal_UACA=-0.7,
    helix_terminal_UACC=-0.6,
    helix_terminal_UACG=-0.7,
    helix_terminal_UACU=-0.5,
    helix_terminal_UAGA=-1.1,
    helix_terminal_UAGC=-0.8,
    helix_terminal_UAGG=-1.2,
    helix_terminal_UAGU=-0.8,
    helix_terminal_UAUA=-0.7,
    helix_terminal_UAUC=-0.6,
    helix_terminal_UAUG=-0.7,
    helix_terminal_UAUU=-0.5,
    helix_terminal_UGAA=-1.0,
    helix_terminal_UGAC=-0.8,
    helix_terminal_UGAG=-1.1,
    helix_terminal_UGAU=-0.8,
    helix_terminal_UGCA=-0.7,
    helix_terminal_UGCC=-0.6,
    helix_terminal_UGCG=-0.7,
    helix_terminal_UGCU=-0.5,
    helix_terminal_UGGA=-0.5,
    helix_terminal_UGGC=-0.8,
    helix_terminal_UGGG=-0.8,
    helix_terminal_UGGU=-0.8,
    helix_terminal_UGUA=-0.7,
    helix_terminal_UGUC=-0.6,
    helix_terminal_UGUG=-0.7,
    helix_terminal_UGUU=-0.5,
    hairpin_03=5.4,
    hairpin_04=5.6,
    hairpin_05=5.7,
    hairpin_06=5.4,
    hairpin_07=6.0,
    hairpin_08=5.5,
    hairpin_09=6.4,
    hairpin_10=6.5,
    hairpin_11=6.6,
    hairpin_12=6.7,
    hairpin_13=6.8,
    hairpin_14=6.9,
    hairpin_15=6.9,
    hairpin_16=7.0,
    hairpin_17=7.1,
    hairpin_18=7.1,
    hairpin_19=7.2,
    hairpin_20=7.2,
    hairpin_21=7.3,
    hairpin_22=7.3,
    hairpin_23=7.4,
    hairpin_24=7.4,
    hairpin_25=7.5,
    hairpin_26=7.5,
    hairpin_27=7.5,
    hairpin_28=7.6,
    hairpin_29=7.6,
    hairpin_30=7.7,
    bulge_01=3.8,
    bulge_02=2.8,
    bulge_03=3.2,
    bulge_04=3.6,
    bulge_05=4.0,
    bulge_06=4.4,
    bulge_07=4.6,
    bulge_08=4.7,
    bulge_09=4.8,
    bulge_10=4.9,
    bulge_11=5.0,
    bulge_12=5.1,
    bulge_13=5.2,
    bulge_14=5.3,
    bulge_15=5.4,
    bulge_16=5.4,
    bulge_17=5.5,
    bulge_18=5.5,
    bulge_19=5.6,
    bulge_20=5.7,
    bulge_21=5.7,
    bulge_22=5.8,
    bulge_23=5.8,
    bulge_24=5.8,
    bulge_25=5.9,
    bulge_26=5.9,
    bulge_27=6.0,
    bulge_28=6.0,
    bulge_29=6.0,
    bulge_30=6.1,
    internal_02=0.5,  # Mathews et al 2004
    internal_03=1.6,  #
    internal_04=1.1,  #
    internal_05=2.1,  #
    internal_06=1.9,
    internal_07=2.1,
    internal_08=2.3,
    internal_09=2.4,
    internal_10=2.5,
    internal_11=2.6,
    internal_12=2.7,
    internal_13=2.8,
    internal_14=2.9,
    internal_15=2.9,
    internal_16=3.0,
    internal_17=3.1,
    internal_18=3.1,
    internal_19=3.2,
    internal_20=3.3,
    internal_21=3.3,
    internal_22=3.4,
    internal_23=3.4,
    internal_24=3.5,
    internal_25=3.5,
    internal_26=3.5,
    internal_27=3.6,
    internal_28=3.6,
    internal_29=3.7,
    internal_30=3.7,
    helix_stacking_AUAU=-0.9,
    helix_stacking_AUCG=-2.2,
    helix_stacking_AUGC=-2.1,
    helix_stacking_AUGU=-0.6,
    helix_stacking_AUUA=-1.1,
    helix_stacking_AUUG=-1.4,
    helix_stacking_CGAU=-2.1,
    helix_stacking_CGCG=-3.3,
    helix_stacking_CGGC=-2.4,
    helix_stacking_CGGU=-1.4,
    helix_stacking_CGUA=-2.1,
    helix_stacking_CGUG=-2.1,
    helix_stacking_GCAU=-2.4,
    helix_stacking_GCCG=-3.4,
    helix_stacking_GCGC=-3.3,
    helix_stacking_GCGU=-1.5,
    helix_stacking_GCUA=-2.2,
    helix_stacking_GCUG=-2.5,
    helix_stacking_GUAU=-1.3,
    helix_stacking_GUCG=-2.5,
    helix_stacking_GUGC=-2.3,
    helix_stacking_GUGU=-0.5,
    helix_stacking_GUUA=-1.4,
    helix_stacking_GUUG=+1.3,
    helix_stacking_UAAU=-1.3,
    helix_stacking_UACG=-2.4,
    helix_stacking_UAGC=-2.1,
    helix_stacking_UAGU=-1.0,
    helix_stacking_UAUA=-0.9,
    helix_stacking_UAUG=-1.3,
    helix_stacking_UGAU=-1.0,
    helix_stacking_UGCG=-1.5,
    helix_stacking_UGGC=-1.4,
    helix_stacking_UGGU=+0.3,
    helix_stacking_UGUA=-0.6,
    helix_stacking_UGUG=-0.5,
)
