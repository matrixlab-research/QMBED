use qmbed::Complex64;
use qmbed::basis::{
    Basis, BosonBasis1D, ExchangeStatistics, GeneralBasis, LatticeSymmetryMap,
    LocalOccupationConstraint, PackedBasis, SpinBasis1D, SpinfulFermionBasis1D,
    SpinlessFermionBasis1D, SymmetryMap, SymmetryReducer, SymmetrySector,
};

#[test]
fn runtime_translation_matches_the_builtin_spin_sector() {
    let sites = 6;
    let parent = SpinBasis1D::builder(sites).up(3).build().unwrap();
    let translation = LatticeSymmetryMap::site_permutation(
        2,
        (0..sites)
            .map(|site| (site + 1) % sites)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let general =
        GeneralBasis::new(parent, SymmetrySector::new().with_map(translation, 1)).unwrap();
    let builtin = SpinBasis1D::builder(sites)
        .up(3)
        .momentum(1)
        .build()
        .unwrap();

    let general_states = (0..general.len())
        .map(|index| general.state(index).unwrap())
        .collect::<Vec<_>>();
    let builtin_states = (0..builtin.len())
        .map(|index| builtin.state(index).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(general_states, builtin_states);

    let packed = PackedBasis::from(general);
    assert_eq!(packed.len(), builtin.len());
    assert_eq!(packed.state(0).unwrap(), builtin.state(0).unwrap());
}

#[test]
fn additive_sector_unions_share_the_normal_basis_and_symmetry_paths() {
    let selected = [0, 2, 4];
    let spin = SpinBasis1D::builder(4)
        .particle_sectors(selected)
        .build()
        .unwrap();
    let fermion = SpinlessFermionBasis1D::builder(4)
        .particle_sectors(selected)
        .build()
        .unwrap();
    assert_eq!(spin.len(), 8);
    assert_eq!(fermion.len(), 8);
    assert_eq!(spin.particle_sectors(), Some(selected.as_slice()));
    assert_eq!(fermion.particle_sectors(), Some(selected.as_slice()));
    for index in 0..spin.len() {
        assert_eq!(spin.state(index).unwrap(), fermion.state(index).unwrap());
    }

    let boson = BosonBasis1D::builder(3, 3)
        .particle_sectors([0, 2])
        .build()
        .unwrap();
    assert_eq!(boson.len(), 7);
    assert_eq!(boson.particle_sectors(), Some([0, 2].as_slice()));

    let translation = LatticeSymmetryMap::site_permutation(2, vec![1, 2, 3, 0]).unwrap();
    let sector_dimensions = (0..4)
        .map(|momentum| {
            GeneralBasis::new(
                SpinBasis1D::builder(4)
                    .particle_sectors(selected)
                    .build()
                    .unwrap(),
                SymmetrySector::new().with_map(translation.clone(), momentum),
            )
            .unwrap()
            .len()
        })
        .sum::<usize>();
    assert_eq!(sector_dimensions, spin.len());
}

#[test]
fn local_binary_species_constraints_compose_with_sector_and_symmetry_reduction() {
    let no_double = LocalOccupationConstraint::new(2, [0, 1, 2]).unwrap();
    let parent = SpinfulFermionBasis1D::builder(3)
        .particles(2, 1)
        .local_occupation_constraint(no_double.clone())
        .build()
        .unwrap();
    assert_eq!(parent.len(), 3);
    assert_eq!(
        parent
            .local_occupation_constraint()
            .unwrap()
            .allowed_local_states(),
        &[0, 1, 2]
    );
    for index in 0..parent.len() {
        assert!(
            no_double
                .accepts_packed_state(parent.state(index).unwrap(), 3)
                .unwrap()
        );
    }

    let destinations = vec![1, 2, 0, 4, 5, 3];
    let translation = LatticeSymmetryMap::site_permutation(2, destinations).unwrap();
    let sector_dimensions = (0..3)
        .map(|momentum| {
            GeneralBasis::new(
                SpinfulFermionBasis1D::builder(3)
                    .particles(2, 1)
                    .local_occupation_constraint(no_double.clone())
                    .build()
                    .unwrap(),
                SymmetrySector::new().with_map(translation.clone(), momentum),
            )
            .unwrap()
            .len()
        })
        .sum::<usize>();
    assert_eq!(sector_dimensions, parent.len());
}

#[test]
fn generated_group_closure_supports_noncommuting_trivial_characters() {
    let sites = 4;
    let translation = LatticeSymmetryMap::site_permutation(2, vec![1, 2, 3, 0]).unwrap();
    let reflection = LatticeSymmetryMap::site_permutation(2, vec![3, 2, 1, 0]).unwrap();
    let dihedral = GeneralBasis::new(
        SpinBasis1D::builder(sites).build().unwrap(),
        SymmetrySector::new()
            .with_map(translation, 0)
            .with_map(reflection, 0),
    )
    .unwrap();

    let representatives = (0..dihedral.len())
        .map(|index| dihedral.state(index).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(representatives, vec![0, 1, 3, 5, 7, 15]);
    let image = dihedral.reduction_image(8).unwrap().unwrap();
    assert_eq!(*image.representative(), 1);
    assert_eq!(image.orbit_size(), 4);
    assert!((image.amplitude().norm() - 0.5).abs() < 1.0e-12);
}

#[test]
fn reducer_queries_orbits_before_basis_materialization() {
    let translation = LatticeSymmetryMap::site_permutation(2, vec![1, 2, 3, 0]).unwrap();
    let reflection = LatticeSymmetryMap::site_permutation(2, vec![3, 2, 1, 0]).unwrap();
    let reducer = SymmetryReducer::new()
        .with_map(translation, 0)
        .with_map(reflection, 0);

    let orbit = reducer.orbit(8).unwrap();
    assert_eq!(*orbit.representative(), 1);
    assert_eq!(orbit.orbit_size(), 4);
    assert!(orbit.is_compatible());
    assert_eq!(orbit.phase(), Some(Complex64::new(1.0, 0.0)));
    assert_eq!(
        orbit.physical_phase_to_representative(),
        Complex64::new(1.0, 0.0)
    );
    assert_eq!(orbit.generator_word(), &[0]);
    assert_eq!(reducer.period_product().unwrap(), 8);

    let basis =
        GeneralBasis::from_reducer(SpinBasis1D::builder(4).build().unwrap(), reducer.clone())
            .unwrap();
    let image = basis.reduction_image(8).unwrap().unwrap();
    assert_eq!(image.phase(), orbit.phase().unwrap());
    assert_eq!(image.orbit_size(), orbit.orbit_size());
    assert_eq!(basis.reducer().generators(), 2);
}

#[test]
fn inconsistent_generator_phases_remove_incompatible_orbits() {
    let translation = LatticeSymmetryMap::site_permutation(2, vec![1, 2, 3, 0]).unwrap();
    let reflection = LatticeSymmetryMap::site_permutation(2, vec![3, 2, 1, 0]).unwrap();
    let incompatible = GeneralBasis::new(
        SpinBasis1D::builder(4).build().unwrap(),
        SymmetrySector::new()
            .with_map(translation, 1)
            .with_map(reflection, 0),
    )
    .unwrap();

    assert!(incompatible.is_empty());
    assert!(incompatible.reduction_image(1).unwrap().is_none());
}

#[test]
fn local_digit_permutations_cover_spin_inversion() {
    let sites = 4;
    let inversion = LatticeSymmetryMap::new(
        2,
        (0..sites).collect::<Vec<_>>(),
        Some(vec![vec![1, 0]; sites]),
        ExchangeStatistics::Distinguishable,
    )
    .unwrap();

    assert_eq!(inversion.period(), 2);
    assert_eq!(
        inversion.apply(0b0011).unwrap(),
        (0b1100, Complex64::new(1.0, 0.0))
    );
    assert_eq!(
        inversion.apply(0b1100).unwrap(),
        (0b0011, Complex64::new(1.0, 0.0))
    );
}

#[test]
fn fermionic_permutations_compute_the_exchange_phase() {
    let swap = LatticeSymmetryMap::fermionic_orbital_permutation(vec![1, 0]).unwrap();

    assert_eq!(swap.period(), 2);
    assert_eq!(swap.apply(0b01).unwrap(), (0b10, Complex64::new(1.0, 0.0)));
    assert_eq!(swap.apply(0b11).unwrap(), (0b11, Complex64::new(-1.0, 0.0)));
}

#[test]
fn malformed_runtime_maps_are_rejected_before_basis_construction() {
    assert!(
        LatticeSymmetryMap::site_permutation(2, vec![0, 0])
            .unwrap_err()
            .to_string()
            .contains("bijection")
    );
    assert!(
        LatticeSymmetryMap::new(
            2,
            vec![1, 0],
            Some(vec![vec![0, 0], vec![0, 1]]),
            ExchangeStatistics::Distinguishable,
        )
        .unwrap_err()
        .to_string()
        .contains("local-state map")
    );
    assert!(
        LatticeSymmetryMap::new(
            2,
            vec![1, 0],
            Some(vec![vec![1, 0], vec![1, 0]]),
            ExchangeStatistics::Fermionic,
        )
        .unwrap_err()
        .to_string()
        .contains("cannot change")
    );
}

#[test]
fn a_valid_empty_symmetry_sector_is_representable() {
    let parent = SpinBasis1D::builder(4).up(0).build().unwrap();
    let translation = LatticeSymmetryMap::site_permutation(2, vec![1, 2, 3, 0]).unwrap();
    let empty = GeneralBasis::new(parent, SymmetrySector::new().with_map(translation, 1)).unwrap();

    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());
    assert_eq!(PackedBasis::from(empty).len(), 0);
}

#[test]
fn exact_u128_radix_capacity_is_accepted() {
    let map = LatticeSymmetryMap::site_permutation(4, (0..64).collect::<Vec<_>>()).unwrap();
    assert_eq!(map.sites(), 64);
    assert_eq!(map.states_per_site(), 4);
    assert_eq!(map.apply(u128::MAX).unwrap().0, u128::MAX);
}
