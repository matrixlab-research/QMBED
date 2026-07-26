"""
Native Julia request types for the shared QMBED Rust exact-diagonalization
core.

All site indices are zero based so an operator request has identical meaning
in Julia, Python, Rust, and the language-neutral C schema.
"""
module QMBED

using JSON3
using LazyArtifacts
using Libdl

export BosonBasis, Coupling, Eigensystem, EigshOptions, LocalOperator
export OpProduct, OperatorSpec, SpinBasis, SpinfulFermionBasis
export SpinlessFermionBasis, eigsh
export IdentityOp, NumberOp, ZOp, RaisingOp, LoweringOp, XOp, YOp

"""
    LocalOperator

Built-in one-site actions recognized by QMBED's universal operator assembler.
The enum values are `IdentityOp`, `NumberOp`, `ZOp`, `RaisingOp`,
`LoweringOp`, `XOp`, and `YOp`.
"""
@enum LocalOperator begin
    IdentityOp
    NumberOp
    ZOp
    RaisingOp
    LoweringOp
    XOp
    YOp
end

const _operator_names = Dict(
    IdentityOp => "identity",
    NumberOp => "number",
    ZOp => "z",
    RaisingOp => "raising",
    LoweringOp => "lowering",
    XOp => "x",
    YOp => "y",
)

"""
    OpProduct(operators; split=nothing)

Ordered product of local actions. `operators` follows the same order as the
zero-based sites in each [`Coupling`](@ref). For a spinful product, `split`
marks the boundary between up- and down-species actions.
"""
struct OpProduct
    operators::Vector{Union{LocalOperator,String}}
    split::Union{Nothing,Int}
end

OpProduct(operators; split=nothing) =
    OpProduct(Union{LocalOperator,String}[operator for operator in operators], split)

"""
    Coupling(coefficient, sites)

One complex coefficient and the zero-based sites on which an
[`OpProduct`](@ref) acts. The number and order of `sites` must match the
product's local actions.
"""
struct Coupling
    coefficient::ComplexF64
    sites::Vector{Int}
end

Coupling(coefficient::Number, sites::AbstractVector{<:Integer}) =
    Coupling(ComplexF64(coefficient), Int[site for site in sites])

"""
    OperatorSpec(product, couplings)

A reusable local [`OpProduct`](@ref) and all couplings multiplying it. The Rust
core validates the arity and assembles the sum in the requested storage format.
"""
struct OperatorSpec
    product::OpProduct
    couplings::Vector{Coupling}
end

OperatorSpec(product::OpProduct, couplings::AbstractVector{<:Coupling}) =
    OperatorSpec(product, Coupling[c for c in couplings])

"""
    BasisSpec

Abstract parent of immutable Julia basis requests accepted by [`eigsh`](@ref).
"""
abstract type BasisSpec end

"""
    SpinBasis(; sites, spin_twice=1, up=nothing, momentum=nothing,
              parity=nothing, pauli=false)

One-dimensional spin basis.

# Keywords
- `sites`: number of lattice sites.
- `spin_twice`: twice the local spin quantum number; `1` means spin one half.
- `up`: fixed total raising-quantum count, or `nothing`.
- `momentum`: translation momentum sector, or `nothing`.
- `parity`: reflection eigenvalue, normally `-1` or `1`.
- `pauli`: use Pauli instead of angular-momentum normalization.
"""
Base.@kwdef struct SpinBasis <: BasisSpec
    sites::Int
    spin_twice::Int = 1
    up::Union{Nothing,Int} = nothing
    momentum::Union{Nothing,Int} = nothing
    parity::Union{Nothing,Int} = nothing
    pauli::Bool = false
end

"""
    BosonBasis(; sites, states_per_site, particles=nothing)

Bosonic lattice basis with local occupations from zero through
`states_per_site - 1`. Set `particles` to restrict the total boson number.
"""
Base.@kwdef struct BosonBasis <: BasisSpec
    sites::Int
    states_per_site::Int
    particles::Union{Nothing,Int} = nothing
end

"""
    SpinlessFermionBasis(; sites, particles=nothing, momentum=nothing)

Spinless-fermion Fock basis with optional fixed particle number and
translation momentum sector.
"""
Base.@kwdef struct SpinlessFermionBasis <: BasisSpec
    sites::Int
    particles::Union{Nothing,Int} = nothing
    momentum::Union{Nothing,Int} = nothing
end

"""
    SpinfulFermionBasis(; sites, particles_up=nothing, particles_down=nothing)

Two-species fermion basis with independent fixed-number sectors for the up and
down species.
"""
Base.@kwdef struct SpinfulFermionBasis <: BasisSpec
    sites::Int
    particles_up::Union{Nothing,Int} = nothing
    particles_down::Union{Nothing,Int} = nothing
end

"""
    EigshOptions(; eigenpairs, target="smallest_algebraic", shift=nothing,
                 krylov_dimension=nothing, tolerance=1e-10,
                 max_iterations=1000, seed=0, eigenvectors=false)

Controls a selected Hermitian eigensolve.

`target` accepts the QMBED spectral target names. Use `target="shift"` together
with `shift` for interior eigenpairs. `krylov_dimension=nothing` selects the
core default; `seed` makes the initial vector deterministic.
"""
Base.@kwdef struct EigshOptions
    eigenpairs::Int
    target::String = "smallest_algebraic"
    shift::Union{Nothing,Float64} = nothing
    krylov_dimension::Union{Nothing,Int} = nothing
    tolerance::Float64 = 1.0e-10
    max_iterations::Int = 1000
    seed::UInt64 = 0
    eigenvectors::Bool = false
end

"""
    Eigensystem

Result of a QMBED eigensolve.

# Fields
- `dimension`: Hilbert-space dimension.
- `eigenvalues`: ordered real eigenvalues or Ritz values.
- `residuals`: norms of `H*v - λ*v`.
- `iterations`: solver iteration count.
- `converged`: whether every requested residual met the tolerance.
- `eigenvectors`: returned vectors, or `nothing` when not requested.
"""
struct Eigensystem
    dimension::Int
    eigenvalues::Vector{Float64}
    residuals::Vector{Float64}
    iterations::Int
    converged::Bool
    eigenvectors::Union{Nothing,Vector{Vector{ComplexF64}}}
end

const _artifacts_toml = normpath(joinpath(@__DIR__, "..", "Artifacts.toml"))

function _artifact_library_path(artifacts_toml=_artifacts_toml)
    isfile(artifacts_toml) || return nothing
    hash = LazyArtifacts.artifact_hash("qmbed_capi", artifacts_toml)
    hash === nothing && return nothing
    LazyArtifacts.ensure_artifact_installed("qmbed_capi", artifacts_toml)
    root = LazyArtifacts.artifact_path(hash)
    library = "libqmbed_capi.$(Libdl.dlext)"
    for candidate in (joinpath(root, "lib", library), joinpath(root, library))
        isfile(candidate) && return candidate
    end
    error("QMBED artifact does not contain $(library)")
end

function _library_path()
    if haskey(ENV, "QMBED_LIBRARY_PATH")
        configured = expanduser(ENV["QMBED_LIBRARY_PATH"])
        isabspath(configured) && return configured
        repository = normpath(joinpath(@__DIR__, "..", "..", ".."))
        return normpath(joinpath(repository, configured))
    end
    artifact = _artifact_library_path()
    artifact === nothing || return artifact
    profile = get(ENV, "QMBED_BUILD_PROFILE", "release")
    source_library = joinpath(
        @__DIR__,
        "..",
        "..",
        "capi",
        "target",
        profile,
        "libqmbed_capi.$(Libdl.dlext)",
    )
    isfile(source_library) && return source_library
    error(
        "QMBED native library not found; reinstall a registered package artifact, " *
        "set QMBED_LIBRARY_PATH, or build bindings/capi with cargo",
    )
end

function _run(request)
    handle = Libdl.dlopen(_library_path())
    run_pointer = Libdl.dlsym(handle, :qmbed_run_json)
    free_pointer = Libdl.dlsym(handle, :qmbed_string_free)
    response_pointer = ccall(run_pointer, Ptr{Cchar}, (Cstring,), JSON3.write(request))
    response_pointer == C_NULL && error("QMBED returned a null response")
    response_text = unsafe_string(response_pointer)
    ccall(free_pointer, Cvoid, (Ptr{Cchar},), response_pointer)
    Libdl.dlclose(handle)
    response = JSON3.read(response_text)
    response.status == "ok" || error(String(response.error))
    response.result
end

function _basis_request(basis::SpinBasis)
    Dict(
        "kind" => "spin",
        "sites" => basis.sites,
        "spin_twice" => basis.spin_twice,
        "up" => basis.up,
        "momentum" => basis.momentum,
        "parity" => basis.parity,
        "pauli" => basis.pauli,
    )
end

function _basis_request(basis::BosonBasis)
    Dict(
        "kind" => "boson",
        "sites" => basis.sites,
        "states_per_site" => basis.states_per_site,
        "particles" => basis.particles,
    )
end

function _basis_request(basis::SpinlessFermionBasis)
    Dict(
        "kind" => "spinless_fermion",
        "sites" => basis.sites,
        "particles" => basis.particles,
        "momentum" => basis.momentum,
    )
end

function _basis_request(basis::SpinfulFermionBasis)
    Dict(
        "kind" => "spinful_fermion",
        "sites" => basis.sites,
        "particles_up" => basis.particles_up,
        "particles_down" => basis.particles_down,
    )
end

function _term_request(term::OperatorSpec)
    local_operators = [
        operator isa LocalOperator ? _operator_names[operator] : operator
        for operator in term.product.operators
    ]
    product = Dict{String,Any}("local" => local_operators)
    isnothing(term.product.split) || (product["split"] = term.product.split)
    Dict(
        "product" => product,
        "couplings" => [
            Dict(
                "coefficient" => [real(coupling.coefficient), imag(coupling.coefficient)],
                "sites" => coupling.sites,
            )
            for coupling in term.couplings
        ],
    )
end

function _solver_request(options::EigshOptions)
    target = options.target == "shift" ?
        Dict("kind" => "shift", "value" => options.shift) :
        Dict("kind" => options.target)
    Dict(
        "eigenpairs" => options.eigenpairs,
        "target" => target,
        "krylov_dimension" => options.krylov_dimension,
        "tolerance" => options.tolerance,
        "max_iterations" => options.max_iterations,
        "seed" => options.seed,
        "eigenvectors" => options.eigenvectors,
    )
end

"""
    eigsh(basis, terms, options; format="csc") -> Eigensystem

Assemble a Hermitian Hamiltonian from typed local terms and compute selected
eigenpairs in the shared Rust core.

# Arguments
- `basis`: a concrete [`BasisSpec`](@ref).
- `terms`: vector of [`OperatorSpec`](@ref) values to sum.
- `options`: spectral target and convergence controls.
- `format`: materialization route, normally `"csc"`; supported alternatives
  are selected by the Rust core.

The returned residuals provide numerical evidence for each eigenpair. Invalid
bases, operator arities, formats, or solver options raise a Julia error
containing the structured QMBED failure.
"""
function eigsh(
    basis::BasisSpec,
    terms::AbstractVector{OperatorSpec},
    options::EigshOptions;
    format="csc",
)
    result = _run(Dict(
        "basis" => _basis_request(basis),
        "terms" => [_term_request(term) for term in terms],
        "format" => format,
        "solver" => _solver_request(options),
    ))
    vectors = hasproperty(result, :eigenvectors) ?
        [
            ComplexF64[ComplexF64(value[1], value[2]) for value in vector]
            for vector in result.eigenvectors
        ] :
        nothing
    Eigensystem(
        Int(result.dimension),
        Float64[value for value in result.eigenvalues],
        Float64[value for value in result.residuals],
        Int(result.iterations),
        Bool(result.converged),
        vectors,
    )
end

end
