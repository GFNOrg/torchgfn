import math

import numpy as np
from MFE_utils import (
    TURNER_FE,
    base_as_int,
    int_as_base,
    pair_int_as_str,
    str_as_pair_int,
    tuple_as_pair_int,
)

_PAIRS = (
    "AU",
    "CG",
    "GC",
    "GU",
    "UA",
    "UG",
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
