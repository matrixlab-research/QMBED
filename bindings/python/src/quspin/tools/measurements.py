"""Compatibility adapters for QuSpin's legacy measurement helpers."""

from __future__ import annotations

import numpy as np

from qmbed._ffi import command


def _ent_entropy(
    system_state,
    basis,
    chain_subsys=None,
    DM=False,
    svd_return_vec=(False, False, False),
    subsys_ordering=True,
    density=True,
    alpha=1.0,
    **options,
):
    if isinstance(system_state, dict):
        if "rho_d" in system_state and "V_rho" in system_state:
            vectors = np.asarray(system_state["V_rho"])
            probabilities = np.asarray(system_state["rho_d"])
            system_state = np.einsum(
                "ji,j,jk->ik",
                vectors,
                probabilities,
                vectors.conj(),
            )
            enforce_pure = False
        elif "V_states" in system_state:
            system_state = system_state["V_states"]
            enforce_pure = True
        else:
            raise ValueError(
                "expecting dictionary with keys ['V_rho','rho_d'] or ['V_states']"
            )
    else:
        enforce_pure = bool(options.pop("enforce_pure", False))

    return_rdm = {
        None: None,
        False: None,
        "chain_subsys": "A",
        "other_subsys": "B",
        "both": "both",
    }.get(DM)
    if DM not in {None, False, "chain_subsys", "other_subsys", "both"}:
        raise TypeError("Unexpected keyword argument for 'DM'!")
    return_singular_values = bool(svd_return_vec[1])

    result = basis.ent_entropy(
        system_state,
        sub_sys_A=chain_subsys,
        density=density,
        alpha=alpha,
        return_rdm=return_rdm,
        return_rdm_EVs=return_singular_values,
        enforce_pure=enforce_pure,
        subsys_ordering=subsys_ordering,
        **options,
    )
    sent_key = "Sent_B" if DM == "other_subsys" else "Sent_A"
    output = dict(result)
    output["Sent"] = result[sent_key]
    if "rdm_A" in result:
        output["DM_chain_subsys"] = result["rdm_A"]
    if "rdm_B" in result:
        output["DM_other_subsys"] = result["rdm_B"]
    if return_singular_values:
        probabilities = np.asarray(result["p_A"])
        if probabilities.ndim == 2:
            probabilities = probabilities.T
        output["lmbda"] = np.sqrt(np.maximum(probabilities, 0.0)).squeeze()
    return output


def ent_entropy(
    system_state,
    basis,
    chain_subsys=None,
    DM=False,
    svd_return_vec=[False, False, False],
    **options,
):
    return _ent_entropy(
        system_state,
        basis,
        chain_subsys=chain_subsys,
        DM=DM,
        svd_return_vec=svd_return_vec,
        **options,
    )


def mean_level_spacing(E, verbose=True):
    del verbose
    values = np.asarray(E, dtype=np.float64).reshape(-1)
    result = command(
        {
            "operation": "mean_level_spacing",
            "eigenvalues": values.tolist(),
        }
    )
    return np.nan if result["value"] is None else float(result["value"])


def _complex_payload(values):
    return [
        [complex(value).real, complex(value).imag]
        for value in np.asarray(values).reshape(-1)
    ]


def _complex_columns(values):
    array = np.asarray(values)
    if array.ndim == 1:
        array = array[:, None]
    return [_complex_payload(array[:, column]) for column in range(array.shape[1])]


def diag_ensemble(
    N,
    system_state,
    E2,
    V2,
    density=True,
    alpha=1.0,
    rho_d=False,
    Obs=False,
    delta_t_Obs=False,
    delta_q_Obs=False,
    Sd_Renyi=False,
    Srdm_Renyi=False,
    Srdm_args=None,
):
    if not isinstance(N, (int, np.integer)) or int(N) <= 0:
        raise TypeError("system size N must be a positive integer")
    N = int(N)
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0.0:
        raise TypeError("Renyi alpha must be real, finite, and nonnegative")
    if (delta_t_Obs or delta_q_Obs) and Obs is False:
        raise TypeError("observable fluctuations require Obs")

    energies = np.asarray(E2, dtype=np.float64).reshape(-1)
    eigenvectors = np.asarray(V2)
    if eigenvectors.shape != (energies.size, energies.size):
        raise ValueError("V2 must contain one square eigenbasis in its columns")

    input_request: dict[str, object]
    mixture_weights = None
    selected_states = None
    normalizations = None
    if isinstance(system_state, dict):
        required = {"V1", "E1", "f_args"}
        missing = sorted(required.difference(system_state))
        if missing:
            raise TypeError(
                "diagonal ensemble state dictionary is missing "
                + ", ".join(missing)
            )
        initial_vectors = np.asarray(system_state["V1"])
        initial_energies = np.asarray(system_state["E1"], dtype=np.float64)
        if initial_vectors.shape != eigenvectors.shape:
            raise ValueError("V1 and V2 must have matching square shapes")
        if initial_energies.shape != energies.shape or np.any(
            np.diff(initial_energies) < 0.0
        ):
            raise ValueError("E1 must be an ordered vector matching V1")
        input_request = {
            "kind": "pure_columns",
            "vectors": _complex_columns(initial_vectors),
        }
        parameters = np.atleast_1d(system_state["f_args"][0])
        distribution = system_state.get(
            "f",
            lambda values, beta: np.exp(
                -float(beta) * (values - values[0])
            ),
        )
        columns = []
        normalizations = []
        normalize_distribution = bool(system_state.get("f_norm", True))
        for parameter in parameters:
            weights = np.asarray(
                distribution(initial_energies, parameter),
                dtype=np.float64,
            ).reshape(-1)
            if (
                weights.size != initial_energies.size
                or np.any(~np.isfinite(weights))
                or np.any(weights < 0.0)
            ):
                raise ValueError(
                    "initial-state distribution must be finite, nonnegative, "
                    "and match E1"
                )
            norm = float(weights.sum())
            if norm <= 0.0:
                raise ValueError("initial-state distribution has zero weight")
            normalizations.append(norm)
            columns.append(weights / norm if normalize_distribution else weights)
        mixture_weights = np.column_stack(columns)
        suffix = "mixed" if "f" in system_state else "thermal"
        if "V1_state" in system_state:
            selected_states = np.asarray(
                system_state["V1_state"],
                dtype=np.intp,
            ).reshape(-1)
            if np.any(selected_states < 0) or np.any(
                selected_states >= energies.size
            ):
                raise ValueError("V1_state index is out of range")
    else:
        initial = np.asarray(system_state)
        if initial.ndim == 1:
            if initial.size != energies.size:
                raise ValueError("initial state and V2 dimensions differ")
            input_request = {
                "kind": "pure",
                "values": _complex_payload(initial),
            }
            suffix = "pure"
        elif initial.ndim == 2 and initial.shape == eigenvectors.shape:
            input_request = {
                "kind": "density",
                "values": _complex_payload(initial),
            }
            suffix = "DM"
        else:
            raise TypeError("system_state must be a vector, density matrix, or dictionary")

    observable_request = None
    if Obs is not False:
        from quspin.operators import _as_operator_expression

        observable_request = _as_operator_expression(Obs)._request()

    response = command(
        {
            "operation": "analyze_diagonal_ensemble",
            "eigenvalues": energies.tolist(),
            "eigenvectors": _complex_columns(eigenvectors),
            "input": input_request,
            "observable": observable_request,
            "alpha": alpha,
            "reconstruct_density": bool(Srdm_Renyi),
        }
    )

    def reduce_columns(values):
        values = np.asarray(values, dtype=np.float64)
        if mixture_weights is None:
            return values[0] if values.size == 1 else values
        return values @ mixture_weights

    def add_metric(name, values, *, subsystem_size=None):
        values = np.asarray(values, dtype=np.float64)
        if density:
            values = values / (
                int(subsystem_size) if subsystem_size is not None else N
            )
        result[f"{name}_{suffix}"] = reduce_columns(values)
        if selected_states is not None:
            result[f"{name}_V1_state"] = values[selected_states]

    result = {}
    if Obs is not False:
        add_metric("Obs", response["observables"])
    if delta_t_Obs or delta_q_Obs:
        add_metric("delta_t_Obs", response["temporal_fluctuations"])
    if delta_q_Obs:
        add_metric("delta_q_Obs", response["quantum_fluctuations"])
    if Sd_Renyi:
        entropy_name = "Sd" if alpha == 1.0 else "Sd_Renyi"
        add_metric(entropy_name, response["diagonal_entropies"])

    if Srdm_Renyi:
        options = dict(Srdm_args or {})
        try:
            basis = options.pop("basis")
        except KeyError as error:
            raise TypeError("Srdm_args must contain basis") from error
        subsystem = options.pop(
            "sub_sys_A",
            options.pop("chain_subsys", None),
        )
        subsystem_size = (
            len(tuple(subsystem))
            if subsystem is not None
            else int(basis.L) // 2
        )
        entropies = []
        for payload in response["density_matrices"]:
            diagonal_density = np.asarray(
                [complex(*value) for value in payload],
                dtype=np.complex128,
            ).reshape(eigenvectors.shape)
            entropies.append(
                basis.ent_entropy(
                    diagonal_density,
                    sub_sys_A=subsystem,
                    alpha=alpha,
                    density=False,
                    **options,
                )["Sent_A"]
            )
        entropy_name = "Srdm" if alpha == 1.0 else "Srdm_Renyi"
        add_metric(
            entropy_name,
            entropies,
            subsystem_size=subsystem_size,
        )

    probabilities = np.asarray(response["probabilities"], dtype=np.float64).T
    if rho_d:
        if selected_states is not None:
            result["rho_d"] = probabilities[:, selected_states]
        elif probabilities.shape[1] == 1:
            result["rho_d"] = probabilities[:, 0]
        else:
            result["rho_d"] = probabilities
    if mixture_weights is not None and not bool(system_state.get("f_norm", True)):
        result["f_norm"] = np.asarray(normalizations)
    return result


def _trajectory_array(states, times):
    from .evolution import ED_state_vs_time

    if isinstance(states, tuple) and len(states) == 3:
        return np.asarray(ED_state_vs_time(*states, times, iterate=False))
    if isinstance(states, np.ndarray):
        trajectory = states
    else:
        trajectory = np.stack(list(states), axis=-1)
    if trajectory.shape[-1] != len(times):
        raise ValueError("state trajectory and time vector lengths differ")
    return trajectory


def _observable_matrix(observable, time):
    if hasattr(observable, "toarray"):
        try:
            return np.asarray(observable.toarray(time=time))
        except TypeError:
            try:
                return np.asarray(observable.toarray(time))
            except TypeError:
                return np.asarray(observable.toarray())
    if hasattr(observable, "todense"):
        return np.asarray(observable.todense())
    return np.asarray(observable)


def _expectation_at(observable, state, time):
    matrix = _observable_matrix(observable, time)
    if state.ndim == 1:
        return np.vdot(state, matrix @ state)
    if state.ndim == 2 and state.shape[0] == state.shape[1]:
        return np.einsum("ij,ji->", state, matrix)
    if state.ndim == 2:
        return np.einsum("ik,ij,jk->k", state.conj(), matrix, state)
    raise ValueError("unsupported state shape in obs_vs_time")


def obs_vs_time(
    psi_t,
    times,
    Obs_dict,
    return_state=False,
    Sent_args=None,
    enforce_pure=False,
    verbose=False,
):
    del enforce_pure, verbose
    times = np.asarray(list(times), dtype=np.float64)
    trajectory = _trajectory_array(psi_t, times)
    result = {}
    for name, observable in Obs_dict.items():
        values = [
            _expectation_at(observable, trajectory[..., index], time)
            for index, time in enumerate(times)
        ]
        result[name] = np.real_if_close(np.asarray(values))

    if Sent_args is not None:
        options = dict(Sent_args)
        basis = options.pop("basis")
        entropy_by_key = {}
        for index in range(times.size):
            entropy = basis.ent_entropy(trajectory[..., index], **options)
            for key, value in entropy.items():
                if key.startswith("Sent_"):
                    entropy_by_key.setdefault(key, []).append(value)
        result["Sent_time"] = {
            key: np.asarray(values)
            for key, values in entropy_by_key.items()
        }
    if return_state:
        result["psi_t"] = trajectory
    return result


from .evolution import ED_state_vs_time
from .misc import KL_div, project_op


__all__ = [
    "diag_ensemble",
    "ED_state_vs_time",
    "ent_entropy",
    "KL_div",
    "mean_level_spacing",
    "obs_vs_time",
    "project_op",
]
