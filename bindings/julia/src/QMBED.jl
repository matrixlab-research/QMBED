module QMBED

using JSON3
using Libdl

export BosonBasis, Coupling, Eigensystem, EigshOptions, LocalOperator
export OpProduct, OperatorSpec, SpinBasis, SpinfulFermionBasis
export SpinlessFermionBasis, eigsh
export IdentityOp, NumberOp, ZOp, RaisingOp, LoweringOp, XOp, YOp

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

struct OpProduct
    operators::Vector{Union{LocalOperator,String}}
    split::Union{Nothing,Int}
end

OpProduct(operators; split=nothing) =
    OpProduct(Union{LocalOperator,String}[operator for operator in operators], split)

struct Coupling
    coefficient::ComplexF64
    sites::Vector{Int}
end

Coupling(coefficient::Number, sites::AbstractVector{<:Integer}) =
    Coupling(ComplexF64(coefficient), Int[site for site in sites])

struct OperatorSpec
    product::OpProduct
    couplings::Vector{Coupling}
end

OperatorSpec(product::OpProduct, couplings::AbstractVector{<:Coupling}) =
    OperatorSpec(product, Coupling[c for c in couplings])

abstract type BasisSpec end

Base.@kwdef struct SpinBasis <: BasisSpec
    sites::Int
    spin_twice::Int = 1
    up::Union{Nothing,Int} = nothing
    momentum::Union{Nothing,Int} = nothing
    parity::Union{Nothing,Int} = nothing
    pauli::Bool = false
end

Base.@kwdef struct BosonBasis <: BasisSpec
    sites::Int
    states_per_site::Int
    particles::Union{Nothing,Int} = nothing
end

Base.@kwdef struct SpinlessFermionBasis <: BasisSpec
    sites::Int
    particles::Union{Nothing,Int} = nothing
    momentum::Union{Nothing,Int} = nothing
end

Base.@kwdef struct SpinfulFermionBasis <: BasisSpec
    sites::Int
    particles_up::Union{Nothing,Int} = nothing
    particles_down::Union{Nothing,Int} = nothing
end

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

struct Eigensystem
    dimension::Int
    eigenvalues::Vector{Float64}
    residuals::Vector{Float64}
    iterations::Int
    converged::Bool
    eigenvectors::Union{Nothing,Vector{Vector{ComplexF64}}}
end

function _library_path()
    if haskey(ENV, "QMBED_LIBRARY_PATH")
        configured = expanduser(ENV["QMBED_LIBRARY_PATH"])
        isabspath(configured) && return configured
        repository = normpath(joinpath(@__DIR__, "..", "..", ".."))
        return normpath(joinpath(repository, configured))
    end
    profile = get(ENV, "QMBED_BUILD_PROFILE", "release")
    joinpath(@__DIR__, "..", "..", "capi", "target", profile, "libqmbed_capi.$(Libdl.dlext)")
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
