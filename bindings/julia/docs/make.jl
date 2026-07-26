using Documenter
using QMBED

DocMeta.setdocmeta!(QMBED, :DocTestSetup, :(using QMBED); recursive=true)

makedocs(
    sitename="QMBED.jl",
    modules=[QMBED],
    checkdocs=:exports,
    warnonly=false,
    format=Documenter.HTML(
        canonical="https://matrixlab-research.github.io/QMBED/julia/api/",
        edit_link="main",
        prettyurls=true,
    ),
    pages=[
        "Julia interface" => "index.md",
        "API reference" => "api.md",
    ],
)
