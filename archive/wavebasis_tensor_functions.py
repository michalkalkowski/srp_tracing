"""
Archived from srp_tracing.grid.WaveBasis on 2026-08-09.

These three methods had no callers anywhere in the srp_tracing repo. Per the
user, they belong conceptually to a different (external) package dealing
with elastic-tensor parameterisation/inversion, not to srp_tracing's ray
tracing pipeline. Removed from srp_tracing/grid.py and kept here so they can
be moved into that package as needed; this file is excluded from the
installed srp_tracing package (see setup.py).

Originally methods of WaveBasis (srp_tracing/grid.py), operating on
`self.c` (a 6x6 Voigt-notation elastic stiffness matrix) and `self.rho`.
Note the index convention: `read_tensor_elements`/`update_tensor_elements`
read/write c[1,1] (not c[0,0]) as "c11" -- this only makes sense under a
specific axis convention (e.g. x as the symmetry axis, y-z as the
propagation plane, transverse isotropy forcing c11==c22 i.e. c[0,0]==c[1,1],
which is exactly what update_tensor_elements sets: `self.c[[0, 1], [0, 1]]
= elements[0]`). This was never documented in the original code -- verify/
document the convention before reusing these.
"""


def set_tensor_range(self, ranges, rho):
    """
    Defines material properties.

    Parameters:
    ---
    ranges: list of lists with ranges of respective elastic constants
            [[c11_min, c11_max], [c13_min, c13_max],[c33_min, c33_max], [c44_min, c44_max]]
    rho: float, density
    """
    self.ranges = ranges
    self.rho = rho


def read_tensor_elements(self):
    c11 = self.c[1, 1]
    c13 = self.c[1, 2]
    c33 = self.c[2, 2]
    c44 = self.c[3, 3]
    return c11, c13, c33, c44


def update_tensor_elements(self, elements):
    self.c[[0, 1], [0, 1]] = elements[0]
    self.c[[0, 1, 2, 2], [2, 2, 0, 1]] = elements[1]
    self.c[2, 2] = elements[2]
    self.c[[3, 4], [3, 4]] = elements[3]
    self.calculate_wavespeeds()
