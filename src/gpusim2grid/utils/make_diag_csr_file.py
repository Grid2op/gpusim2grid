# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
from scipy.sparse import csr_matrix


def make_diag_csr(Ybus, nb_solve):
    size = Ybus.shape[0]
    indptr = Ybus.indptr
    indices = Ybus.indices
    last_indptr = indptr[-1]

    data_ = np.concat([Ybus.data for _ in range(nb_solve)])
    indices_ = np.concatenate([i * size + indices for i in range(nb_solve)])
    indptr_ = np.concatenate([np.zeros(shape=(1,), dtype=int)] + [i * last_indptr + indptr[1:] for i in range(nb_solve)])
    Ybus_ = csr_matrix((data_, indices_, indptr_), shape=(size * nb_solve, size * nb_solve))
    return Ybus_