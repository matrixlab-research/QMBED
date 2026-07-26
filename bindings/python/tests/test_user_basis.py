import numpy as np
from numba import carray, cfunc

from quspin.basis.user import (
    op_sig_64,
    pre_check_state_sig_64,
    user_basis,
)
from quspin.operators import hamiltonian


@cfunc(op_sig_64)
def _number_operator_64(result_pointer, operator, site, sites, arguments):
    result = carray(result_pointer, 1)[0]
    encoded_site = sites - site - 1
    if operator == 110:
        result.matrix_ele *= (result.state >> encoded_site) & 1
        return 0
    return -1


@cfunc(pre_check_state_sig_64)
def _keep_odd_states_64(state, sites, arguments):
    return state & 1


def test_user_basis_64_callbacks_and_precheck_use_native_runtime_basis():
    basis = user_basis(
        np.uint64,
        3,
        {
            "op": _number_operator_64,
            "op_args": np.asarray([], dtype=np.uint64),
        },
        allowed_ops={"n"},
        pre_check_state=(
            _keep_odd_states_64,
            np.asarray([], dtype=np.uint64),
        ),
    )
    np.testing.assert_array_equal(
        basis.states,
        np.asarray([7, 5, 3, 1], dtype=np.uint64),
    )
    operator = hamiltonian(
        [["n", [[1.0, 2]]]],
        [],
        basis=basis,
        dtype=np.float64,
        check_symm=False,
        check_herm=False,
        check_pcon=False,
    )
    np.testing.assert_allclose(operator.toarray(), np.eye(4))


def test_user_basis_accepts_general_unit_modulus_exchange_phases():
    basis = user_basis(
        np.uint64,
        2,
        {
            "op": _number_operator_64,
            "op_args": np.asarray([], dtype=np.uint64),
        },
        allowed_ops={"n"},
        noncommuting_bits=[(np.arange(2), 1.0j)],
    )
    state = np.zeros(basis.Ns, dtype=np.complex128)
    state[basis.index(2)] = 1.0 / np.sqrt(2.0)
    state[basis.index(3)] = 1.0 / np.sqrt(2.0)

    density = basis.partial_trace(
        state,
        sub_sys_A=[1],
        return_rdm="A",
    )
    np.testing.assert_allclose(np.diag(density), [0.5, 0.5])
    np.testing.assert_allclose(np.real(density[0, 1]), 0.0, atol=1.0e-12)
    np.testing.assert_allclose(abs(np.imag(density[0, 1])), 0.5, atol=1.0e-12)

    with np.testing.assert_raises_regex(ValueError, "unit magnitude"):
        user_basis(
            np.uint64,
            2,
            {
                "op": _number_operator_64,
                "op_args": np.asarray([], dtype=np.uint64),
            },
            allowed_ops={"n"},
            noncommuting_bits=[(np.arange(2), 2.0)],
        )
