import importlib


PUBLIC_SYMBOLS = {
    "quspin.basis": {
        "spin_basis_1d",
        "boson_basis_1d",
        "spinless_fermion_basis_1d",
        "spinful_fermion_basis_1d",
        "spin_basis_general",
        "boson_basis_general",
        "spinless_fermion_basis_general",
        "spinful_fermion_basis_general",
        "user_basis",
        "tensor_basis",
        "photon_basis",
        "coherent_state",
        "photon_Hspace_dim",
        "basis_ones",
        "basis_zeros",
        "get_basis_type",
        "python_int_to_basis_int",
        "basis_int_to_python_int",
        "bitwise_not",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "bitwise_leftshift",
        "bitwise_rightshift",
    },
    "quspin.operators": {
        "hamiltonian",
        "quantum_operator",
        "exp_op",
        "quantum_LinearOperator",
        "commutator",
        "anti_commutator",
        "ishamiltonian",
        "isquantum_operator",
        "isexp_op",
        "isquantum_LinearOperator",
        "save_zip",
        "load_zip",
    },
    "quspin.tools.evolution": {
        "ED_state_vs_time",
        "evolve",
        "ExpmMultiplyParallel",
        "expm_multiply_parallel",
    },
    "quspin.tools.lanczos": {
        "lanczos_full",
        "lanczos_iter",
        "expm_lanczos",
        "lin_comb_Q_T",
        "LTLM_static_iteration",
        "FTLM_static_iteration",
    },
    "quspin.tools.Floquet": {"Floquet", "Floquet_t_vec"},
    "quspin.tools.measurements": {
        "ent_entropy",
        "diag_ensemble",
        "obs_vs_time",
    },
    "quspin.tools.block_tools": {"block_ops", "block_diag_hamiltonian"},
    "quspin.tools.misc": {
        "matvec",
        "get_matvec_function",
        "mean_level_spacing",
        "project_op",
        "KL_div",
        "ints_to_array",
        "array_to_ints",
    },
    "quspin.basis.transformations": {"square_lattice_trans"},
}


def test_documented_quspin_public_symbols_are_importable():
    for module_name, expected in PUBLIC_SYMBOLS.items():
        module = importlib.import_module(module_name)
        missing = expected.difference(dir(module))
        assert not missing, f"{module_name} is missing {sorted(missing)}"


def test_version_compatibility_module_is_importable():
    from quspin import __version__
    from quspin._version import __version__ as internal_version

    assert __version__ == internal_version == "1.0.1"
