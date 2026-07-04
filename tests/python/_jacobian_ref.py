"""CPU reference Newton-Raphson Jacobian, used only by the test suite.

Extracted from the (removed) pure-Python reference solver so that the adjoint
regression test has a self-contained CPU reference. The production CPU oracle is
lightsim2grid; this helper exists purely to validate ``AcPfNrSession.solve_JT``.
"""
import numpy as np
from scipy.sparse import csr_matrix, diags, bmat


def compute_jacobian(V, Ybus, pvpq, pq):
    """Assemble the NR Jacobian using the MATPOWER sparse formulation.

    Shape: (|pvpq| + |pq|) x (|pvpq| + |pq|), real CSR.

        dS/dVa = j * diagV * (diagIbus - Ybus * diagV)*
        dS/dVm =     diagV * (Ybus * diagVnorm)*  +  diagIbus* * diagVnorm
    """
    n = len(V)
    Ibus = Ybus @ V

    # Diagonal sparse matrices (n x n, complex)
    diagV = diags(V, shape=(n, n), format='csr')
    diagIbus = diags(Ibus, shape=(n, n), format='csr')
    diagVnorm = diags(V / np.abs(V), shape=(n, n), format='csr')

    # Full n x n complex sparse building blocks.
    # .conj() on a scipy sparse matrix conjugates only the stored non-zero values.
    dS_dVa = 1j * diagV @ (diagIbus - Ybus @ diagV).conj()
    dS_dVm = diagV @ (Ybus @ diagVnorm).conj() + diagIbus.conj() @ diagVnorm

    def _sub(M, rows, cols):
        return csr_matrix(M[np.ix_(rows, cols)])

    J11 = _sub(dS_dVa, pvpq, pvpq).real   # dP/dtheta  (|pvpq| x |pvpq|)
    J12 = _sub(dS_dVm, pvpq, pq).real     # dP/d|V|    (|pvpq| x |pq|  )
    J21 = _sub(dS_dVa, pq, pvpq).imag     # dQ/dtheta  (|pq|   x |pvpq|)
    J22 = _sub(dS_dVm, pq, pq).imag       # dQ/d|V|    (|pq|   x |pq|  )

    return bmat([[J11, J12], [J21, J22]], format='csr')
