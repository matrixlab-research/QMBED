# Python native API

The objects below are thin, typed request builders. Numerical work begins only
when an operation such as `eigsh` enters the shared Rust library.

::: qmbed
    options:
      members:
        - QmbedError
        - LocalOperator
        - OpProduct
        - Coupling
        - OperatorSpec
        - BasisSpec
        - SpinBasis
        - BosonBasis
        - SpinlessFermionBasis
        - SpinfulFermionBasis
        - EigshOptions
        - Eigensystem
        - eigsh
