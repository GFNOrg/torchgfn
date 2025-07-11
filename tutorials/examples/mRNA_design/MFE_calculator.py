import math
from typing import Union

import numpy as np
from draw_rna import ipynb_draw


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


class RNASolutionBase:
    """An RNA folding solution and working data

    An instance holds the working data used when solving
    an RNA folding problem, and afterwards to generate.
    various expressions of the solution.
    """

    def __init__(self, rna: str):
        """Create a holder for the RNA folding solution

        The solution is read from the working arrays allocated here.

        U[i,j] is the minimum energy of [i,j] when j is not paired with i.
        V[i,j] is the minimum energy of [i,j] closed by (i,j). It is based
            on V[i+1,j-1] (helix stacking) or U[i+1,j-1] (loop closure)
            whichever gives the lower result, and is math.inf if (i,j) is
            inadmissible as a pair.
        K[i,j] is the nucleotide paired with j in computing U[i,j] or -1
            if the minimum U[i,j] is obtained by leaving j unpaired.
        M[i,j] is 1 if V[i,j] is based on helix stacking, is 0 if V[i,j] is
            based on loop closure, and is -1 if (i,j) is inadmissible. This
            is not a question of V[i+1,j-1] and U[i+1,j-1] values only.
        """
        # Allow mixed case bases and space character initially
        if junk := rna.strip("ACGUacgu "):
            raise ValueError(f"RNA bases must be from {{ACGU}} found {junk!r}.")
        # Remove spaces for computation
        self.rna = rna = "".join(rna.split())
        self.N = N = len(rna)
        # We use integer values to represent the 4 bases
        self.irna = np.fromiter(
            map(base_as_int, rna.upper()), dtype=np.int8, count=len(rna)
        )
        # Storage for partial results
        self.V = np.zeros((N, N), dtype=np.float32)
        self.U = np.zeros((N, N), dtype=np.float32)
        self.K = np.zeros((N, N), dtype=np.int16)
        self.M = np.zeros((N, N), dtype=np.int8)

    def getU(self, i, j) -> float:
        return float(self.U[i, j])

    def getV(self, i, j) -> float:
        return float(self.V[i, j])

    def getK(self, i, j) -> int:
        return int(self.K[i, j])

    def getM(self, i, j) -> int:
        return int(self.M[i, j])

    def energy(self):
        """The free energy of the minimum energy solution"""
        n = self.N - 1
        return min(self.U[0, n], self.V[0, n])


class RNASolutionBase2(RNASolutionBase):

    def get_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get references to the K, U and V arrays"""
        return self.K, self.U, self.V

    def branches(self, i: int, j: int) -> list[tuple]:
        """Enumerate the branches in the open interval [i,j].

        When we consider the energy freed by potential pair (i-1,j+1),
        we must understand what open structure it "closes". It is enough
        to list the branches (helices) chosen when forming the open
        conformation of [i,j], each identified by its initial pair, in
        ascending order of first member.

        The number of branches distingishes 0: hairpin, 1: internal or bulge
        loop, and >1: multiloop, while in case 1 the differences between the
        ends of the proposed closing and those of the single base pair
        distinguish amongst bulges, interior and muylti-branch loops.
        A single branch does not signify helix continuation, since that is
        a closed structure.
        """
        assert 0 <= i
        assert j < self.N
        K, U, V = self.get_arrays()
        branches = []
        j0 = j
        while j > i:
            # Consider the sub-range [i,j] of [i,j0]
            k = K[i, j]
            if k >= 0:
                # We based the energy on [i,k-1] and V[k,j].
                # Add (k,j) as a branch on the loop
                assert k != j
                branches.insert(0, (k, j))
                # Step left to just before the branch
                j = k - 1
            elif j < j0 and V[i, j] < U[i, j]:
                # We based the energy of the sub-range [i,j] on V[i,j].
                # Note we exclude j == j0 as it makes [i,j] not a loop.
                # Add (i,j) as a branch on the loop
                branches.insert(0, (i, j))
                # This branch is the last.
                j = i
            else:
                # j is not paired: step left
                j -= 1

        return branches


class RNASolution(RNASolutionBase2):
    def walk_range_ex(self, i: int, j: int, closed):
        """Walk the range [i,j] inclusive specifying open or closed.

        We use this when backtracking the energy calculation to discover
        the lowest energy structure. Nodes (the data for a range) in our
        algorithm offer both closed (j pairs with a) and open (j pairs
        with something else or is unpaired) energy, and a subsequent
        nodes choose between them in offering their own energies. When
        backtracking, determining the structure of [i,j] in the minimum
        energy configuration may be informed by knowledge of that choice,
        provided through the 'closed' argument.
        """
        # We walk a leftmost depth first recursion
        # print(f"walk_range_ex({i}, {j}, closed={closed})")
        assert 0 <= i <= j < self.N
        if j == i:
            # There is one (unpaired) base in the range
            # print(f"{i} is unpaired")
            assert not closed
            yield i, -1
        elif closed:
            # We based the final energy on V[i, j].
            # print(f"{j} pairs with {i}")
            yield i, j
            if j - i > 1:
                # Process interior with [i,j]'s choice of closed/open
                c = self.getM(i, j) == 1
                yield from self.walk_range_ex(i + 1, j - 1, c)
        elif (k := self.getK(i, j)) >= 0:
            # We based the final energy on [i,k-1] and V[k,j].
            # print(f"{j} pairs with {k}, after [{i},{k - 1}]")
            assert k > i
            # We used the lesser of U[i,k-1] and V[i,k-1].
            yield from self.walk_range(i, k - 1)
            yield k, j
            if j - k > 1:
                # We definitely used V[k,j]
                yield from self.walk_range_ex(k, j, True)
        else:
            # We based the final energy on j unpaired
            # print(f"{j} is unpaired")
            yield from self.walk_range(i, j - 1)
            yield j, -1

    def walk_range(self, i, j):
        """Walk the range [i,j] inclusive using min(U,V).

        We use this when backtracking the energy calculation to discover
        the lowest energy structure, in contexts where we know the subsequent
        energy calculation will have chosen the lesser of the closed (j pairs
        with i) and open (j pairs with something else or is unpaired) energy.
        """
        c = self.getV(i, j) < self.getU(i, j)
        yield from self.walk_range_ex(i, j, c)

    def as_dots(self):
        """Dot-bracket form of the fold"""
        D, L, R = ".()"
        dots = [""] * self.N
        for a, b in self.walk_range(0, self.N - 1):
            if b < 0:
                dots[a] = D
            else:
                dots[a] = L
                dots[b] = R
        return "".join(dots)


class RNAFolderBase:
    """An RNA folding algorithm and parameters

    An instance holds all the configuration necessary to
    solve an RNA folding problem. Note that energies are
    the free-energy (a more negative value for the more
    favourable pairings).
    """

    # Internal loop sizes 0..MAX_INTERNAL inclusive are supported.
    MAX_INTERNAL = 20
    # Bulge loop sizes 0..MAX_BULGE inclusive are supported.
    MAX_BULGE = 20
    # Hairpin loop sizes 0..MAX_HAIRPIN inclusive are supported.
    MAX_HAIRPIN = 20

    def __init__(self, loop_min=0, energies=None):
        self.lmin = loop_min
        if energies is None:
            energies = TURNER_FE
        self.non_pairing = self._special_lookups(energies)
        self._stack_fe, self._term_fe = self._helix_lookups(energies)
        self._internal_fe, self._bulge_fe, self._hairpin_fe = self._loop_lookups(
            energies
        )

    @staticmethod
    def _special_lookups(energies: dict, dflt=None):
        """Get misc special free energy parameters"""
        if dflt is None:
            dflt = energies.get("non_pairing", math.inf)
        return dflt

    @staticmethod
    def _helix_lookups(energies: dict):
        """Create look-up tables for free energy in helices"""
        NPAIRS = len(_PAIRS)  # 6
        NBASES = 4
        # Free energies relating to helix building or termination
        stack = np.zeros((NPAIRS, NPAIRS), dtype=np.float32)
        hterm = np.zeros((NPAIRS, NBASES, NBASES), dtype=np.float32)
        for r in range(NPAIRS):
            # r is a pair in a helix
            rs = pair_int_as_str(r)
            # Default free energy for r (used if no stacking/terminating FE)
            dflt_fe = energies.get(f"base_pair_{rs}")
            if dflt_fe is None:
                dflt_fe = energies.get(f"base_pair_{rs[1]}{rs[0]}", 0.0)
            # s is the following pair in the helix
            for s in range(NPAIRS):
                ss = pair_int_as_str(s)
                stack[r, s] = energies.get(f"helix_stacking_{rs}{ss}", dflt_fe)
            # x and y are un-paired bases terminating the helix
            for x in range(NBASES):
                xs = int_as_base(x)
                for y in range(NBASES):
                    ys = int_as_base(y)
                    hterm[r, x, y] = energies.get(
                        f"helix_terminal_{rs}{xs}{ys}", dflt_fe
                    )
        # Free energies relating to loops
        return stack, hterm

    @staticmethod
    def _interpolated_table(
        energies: dict, size: int, key: str, fmt: str = "02d"
    ) -> np.ndarray:
        """Read values key01, key02, ... to array interpolating missing values.

        We use this to encode e.g. the hairpin loop energies in a table
        we can use incrementally during minimisation.
        """
        table = np.zeros(size, dtype=np.float32)
        m = -1
        inc = 0.0
        for n in range(size):
            # If a value is not given, interpolate.
            hn = energies.get(key + format(n, fmt))
            if hn is not None:
                table[n] = hn
                if n > m + 1:
                    # Interpolate missing between m and n
                    hm = table[m] if m >= 0 else 0.0
                    inc = (hn - hm) / (n - m)
                    for i in range(1, n - m):
                        table[m + i] = hm + inc * i
                m = n
            elif n > 0:
                # Extrapolate. (We may return if can interpolate.)
                table[n] = table[n - 1] + inc
            else:
                # First in table. (We may return if can interpolate.)
                table[0] = 0.0

        return table

    @staticmethod
    def _loop_lookups(energies: dict):
        """Create look-up tables for loop free energy in loops"""
        # Free energy vlues when building a hairpin or other loop
        # Turner provides an energy penalty for each loop size.
        i = RNAFolderBase._interpolated_table(
            energies, RNAFolderBase.MAX_INTERNAL + 1, "internal_"
        )
        b = RNAFolderBase._interpolated_table(
            energies, RNAFolderBase.MAX_BULGE + 1, "bulge_"
        )
        h = RNAFolderBase._interpolated_table(
            energies, RNAFolderBase.MAX_HAIRPIN + 1, "hairpin_"
        )
        return i, b, h

    def print_stack_fe(self, rp: str, sp: str):
        r = str_as_pair_int(rp)
        s = str_as_pair_int(sp)
        e = self._stack_fe[r, s]
        print(f"Free energy of {rp} stacked on {sp} (helix_stacking_{rp}{sp}) ={e:6.1f}")

    def print_term_fe(self, rp: str, an: str, bn: str):
        r = str_as_pair_int(rp)
        a = base_as_int(an)
        b = base_as_int(bn)
        e = self._term_fe[r, a, b]
        print(
            f"Free energy closing with {rp} a loop {an}...{bn} (helix_terminal_{rp}{an}{bn}) ={e:6.1f}"
        )

    def print_hairpin_fe(self, n: int):
        e = self._hairpin_fe[n]
        print(
            f"Energy penalty when closing a hairpin loop of size {n} (hairpin_{n:02d}) ={e:6.1f}"
        )


class RNAFolderBase2(RNAFolderBase):

    def closed_energy(self, soln: RNASolution, i: int, j: int) -> tuple[float, int]:
        """Minimum energy and mode of [i, j] closed by pairing (i,j).

        If the pair is forbidden (not AU, CG, GC, GU, UA or UG) the method
        returns the self.non_pairing energy (normally math.inf), and the
        returned mode is -1. Otherwise, the method compares the minimum energy
        obtained by pairing (i,j) across [i+1, j-1] considered as open and
        closed, and returns the lower energy and mode (0=open, 1=closed).
        """
        # Make the working arrays local
        K, U, V = soln.get_arrays()

        vmin = self.non_pairing
        mode = -1

        # Get the (int) bases at i, j: the ends of the range
        irna = soln.irna
        ri = irna[i]
        rj = irna[j]
        if (r := tuple_as_pair_int(ri, rj)) < 0:
            # (i,j) cannot be a pair.
            return vmin, mode

        # Get the (int) bases at i+1, j-1: the ends of the enclosed range
        si = irna[i1 := i + 1]
        sj = irna[j1 := j - 1]

        # Hypothesise that (i,j) closes a loop (begins a helix), assuming that
        # the minimum energy open conformation defines the loop structure.
        # Begin with the energy from closure over [i1,j1]:
        v1 = self._term_fe[r, si, sj]

        # TODO: fully consider alternative open interior folds.
        # The minimum energy open conformation may not lead to the least energy.

        # The number n of unpaired nucleotides in the loop is critical.
        # We begin with a list of the branches (helices) in the loop.
        b = soln.branches(i1, j1)

        if (nb := len(b)) == 0:
            # (i,j) closes the hairpin loop [i1,j1] of size n.
            n = j1 - i
            # The energy (penalty) is in the hairpin loop table:
            e = self._hairpin_fe

        elif nb == 1:
            # (i,j) closes the internal or bulge loop [i1,j1].
            # The number of unpaired nucleotides outside the branch is n.
            p, q = b[0]
            n = j1 - i1 - (q - p)
            if p == i1 or q == j1:
                # The energy (penalty) is in the bulge loop table:
                e = self._bulge_fe
            else:
                # The energy (penalty) is in the internal loop table:
                e = self._internal_fe
            # The branch (p,q) contributes energy:
            v1 += V[p, q]

        else:
            # (i,j) closes the multi-branch loop [i1,j1].
            # The number of unpaired nucleotides outside any branch is n.
            n = j - i1 - nb
            for p, q in b:
                # Nucleotides under the branch (p,q) do not count in n.
                n -= q - p
                # The branch (p,q) contributes energy:
                v1 += V[p, q]
            # The energy (penalty) is in the internal loop table:
            e = self._internal_fe

        # Consider as loop closure only if not stacking.
        if n > 0 or nb != 1:
            v2 = e[n if n < len(e) else -1]
            vmin = v1 + v2
            mode = 0  # interior is open (if loop wins)

        # Hypothesise that (i,j) stacks on helix [i1,j1].
        if (s := tuple_as_pair_int(si, sj)) >= 0:
            # (i1,j1) can in fact pair. Suppose [i1,j1] to be closed.
            v1 = self._stack_fe[r, s]
            v2 = V[i1, j1] if j1 > i1 else 0.0
            if (v := v1 + v2) < vmin:
                vmin = v
                mode = 1  # interior is closed (stacking wins)

        return vmin, mode

    def open_energy(self, soln: RNASolution, i: int, j: int) -> tuple[float, int]:
        """Minimum energy of [i, j] without pairing (i,j).

        The method seeks a lowest energy for [i,j], considering j unpaired
        or paired with any interior nucleotide k, where i<k<j.
        """
        # Make the working arrays local
        K, U, V = soln.get_arrays()

        # Consider several hypotheses about the addition of j
        # to the range [i,j-1] without pairing them. In each hypothesis
        # where umin is lowered, kmin remembers the pairing of j.
        j1 = j - 1

        # Hypothesise that j dangles right from structure [i,j-1]
        u0 = U[i, j1]
        u1 = V[i, j1]
        umin = min(u0, u1)
        kmin = -1

        # Hypothesise that j pairs with k and [i,k-1] is optimally folded.
        for k in range(i + 1, j):
            u0 = U[i, k - 1]
            u1 = V[i, k - 1]
            u2 = V[k, j]
            if (u := min(u0, u1) + u2) < umin:
                umin = u
                kmin = k

        return umin, kmin


class RNAFolder(RNAFolderBase2):
    def solve(self, rna: str):
        # Hold the working arrays and RNA string in an object.
        soln = RNASolution(rna)
        N = soln.N
        K, U, V = soln.get_arrays()
        M = soln.M

        # Subsequences up to smallest loop size contain no pairs
        for p in range(0, self.lmin + 1):
            for i in range(N - p):
                j = i + p
                V[i, j] = math.inf
                K[i, j] = -1
                M[i, j] = -1

        for p in range(self.lmin + 1, N):
            # in pass p, we determine open and closed free energy for
            # subsequences of length p. (i,j) moves diagonally down and
            # right in i,j-space with j - i == p.
            for i in range(N - p):
                j = i + p

                # Compute the energy with (i,j) paired if admissible.
                vmin, mode = self.closed_energy(soln, i, j)

                # Compute the energy with (i,j) unpaired
                umin, k = self.open_energy(soln, i, j)

                # Store scores for structure [i,j]
                K[i, j] = k
                M[i, j] = mode
                U[i, j] = umin
                V[i, j] = vmin

        return soln


COUNT_PSEUDO_FE = dict(base_pair_AU=-1, base_pair_CG=-1, base_pair_GU=-1)


def print_workings(s: RNASolution):
    """Tabulate the scores in the solution"""
    rna = s.rna
    N = len(rna)
    print("U, K:")
    print("     " + "".join(f"{j:5d} " for j in range(N)))
    print("     " + " -----" * N)
    for i in range(N):
        print(f"{i:3d} :", end="")
        for j in range(N):
            if j < i:
                u = s.getU(j, i)
                print("    . " if u == 0.0 else f"{u:6.1f}", end="")
            elif j > i:
                k = s.getK(i, j)
                print("    . " if k < 0 else f"{k:5d} ", end="")
            else:
                print(f"{'=' + rna[i]:>5s} ", end="")
        print()

    print("V, M:")
    for i in range(N):
        print(f"{i:3d} :", end="")
        for j in range(N):
            if j < i:
                v = s.getV(j, i)
                print("    . " if math.isinf(v) else f"{v:6.1f}", end="")
            elif j > i:
                m = s.getM(i, j)
                print("    . " if m < 0 else f"{m:5d} ", end="")
            else:
                print(f"{'=' + rna[i]:>5s} ", end="")
        print()


def print_ruler(rna: str):
    N = len(rna)
    for i in range(0, N, 10):
        print(f"{i // 10:<10d}", end="")
    print()
    for i in range(N):
        print(f"{i % 10:1d}", end="")
    print()


def test_graph(rna: str, energies=None, loop_min=4):
    # Compute a structure
    f = RNAFolder(energies=energies, loop_min=loop_min)

    s = f.solve(rna)
    rna = s.rna

    structure = s.as_dots()
    energy = s.energy()

    print_ruler(rna)
    print(rna)
    print(structure)
    print(f"Free energy: {energy:8.1f}")

    # Draw using draw_rna
    ipynb_draw.draw_struct(rna, structure)
