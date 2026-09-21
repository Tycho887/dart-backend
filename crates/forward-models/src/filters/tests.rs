use super::*;

const LINES: [&str; 2] = [
    "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
];

fn filter(kind: &str, noise: State, gate: Option<f64>) -> DopplerFilter {
    let tle = TLE::load_2line(LINES[0], LINES[1]).unwrap();
    let epoch = tle.epoch.as_unixtime();
    DopplerFilter::new(
        tle,
        ITRFCoord::from_geodetic_deg(63.0, 10.0, 0.0),
        400e6,
        epoch,
        State::zeros(),
        Covariance::from_diag(&State::from_array([4.0, 100.0, 1e8])),
        noise,
        kind,
        gate,
    )
    .unwrap()
}

fn assert_same(actual: &FilterState, expected: &FilterState) {
    assert_eq!(actual.epoch_unix_s, expected.epoch_unix_s);
    assert_eq!(actual.state, expected.state);
    assert_eq!(actual.covariance, expected.covariance);
}

#[test]
fn all_backends_match_linear_kalman_reference() {
    let h = Sensitivity::new([[1.0, 2.0, -1.0]]);
    let initial = Covariance::new([[1.0, 0.2, -0.1], [0.2, 2.0, 0.3], [-0.1, 0.3, 3.0]]);
    let p = initial + Covariance::eye() * 0.1;
    let s = (h * p * h.transpose())[(0, 0)] + 0.5;
    let expected_x = p * h.transpose() * Measurement::from_array([2.0 / s]);
    let expected_p = p - p * h.transpose() * h * p / s;
    for kind in ["ukf", "srukf", "ekf"] {
        let mut backend = Backend::new(kind, State::zeros(), initial).unwrap();
        backend.predict(&(Covariance::eye() * 0.1)).unwrap();
        let nis = backend
            .update(&Measurement::from_array([2.0]), |x| h * *x, h, 0.5, None)
            .unwrap()
            .unwrap();
        assert!((nis - 4.0 / s).abs() < 1e-12, "{kind}");
        assert!((backend.state() - expected_x).norm() < 1e-12, "{kind}");
        assert!(
            (backend.covariance() - expected_p).norm_inf() < 1e-12,
            "{kind}"
        );
    }
}

#[test]
fn measurement_and_derivatives_match_shared_batch_fixture() {
    let filter = filter("ekf", State::zeros(), None);
    let fixture = include_str!("../../tests/fixtures/doppler_filter.csv");
    for line in fixture.lines().skip(1) {
        let values: Vec<f64> = line.split(',').map(|v| v.parse().unwrap()).collect();
        let state = State::from_array([values[1], values[2], values[3]]);
        let measurement = filter.measurement(values[0], &state).unwrap();
        let sensitivity = filter.sensitivity(values[0], &state).unwrap();
        assert!((measurement[0] - values[4]).abs() < 1e-8);
        assert!((sensitivity[(0, 0)] - values[5]).abs() < 1e-8);
        assert_eq!(sensitivity[(0, 1)], values[6]);
        assert!((sensitivity[(0, 2)] - values[7]).abs() < 1e-14);
        // Independent difference in the nuisance coordinate itself.
        let mut plus = state;
        let mut minus = state;
        plus[0] += 0.05;
        minus[0] -= 0.05;
        let difference = (filter.measurement(values[0], &plus).unwrap()[0]
            - filter.measurement(values[0], &minus).unwrap()[0])
            / 0.1;
        assert!((sensitivity[(0, 0)] - difference).abs() < 2e-3);
    }
}

#[test]
fn prediction_and_rejection_retain_expected_prior() {
    let rates = State::from_array([0.01, 0.2, 3.0]);
    for kind in ["ukf", "srukf", "ekf"] {
        let mut filter = filter(kind, rates, Some(6.63));
        let before = filter.get_state();
        let after = filter.predict(before.epoch_unix_s + 10.0).unwrap();
        assert!((after.state - before.state).norm() < 1e-10);
        let expected = before.covariance + Covariance::from_diag(&(rates * 10.0));
        assert!((after.covariance - expected).norm_inf() < 1e-6);
        let unchanged = filter.predict(after.epoch_unix_s).unwrap();
        assert_same(&after, &unchanged);
        let rejected = filter.update(after.epoch_unix_s + 2.0, 1e9, 1.0).unwrap();
        assert_eq!(rejected, None);
        let rejected_state = filter.get_state();
        assert_eq!(rejected_state.epoch_unix_s, after.epoch_unix_s + 2.0);
        let expected = after.covariance + Covariance::from_diag(&(rates * 2.0));
        assert!((rejected_state.covariance - expected).norm_inf() < 1e-6);
    }
}

#[test]
fn failures_do_not_commit_prediction_or_update() {
    for kind in ["ukf", "srukf", "ekf"] {
        let mut filter = filter(kind, State::from_array([0.01, 0.1, 1.0]), None);
        let before = filter.get_state();
        assert!(
            filter
                .update(before.epoch_unix_s + 10.0, 0.0, -1.0)
                .is_err()
        );
        assert_same(&filter.get_state(), &before);
        assert!(filter.predict(before.epoch_unix_s - 1.0).is_err());
        assert_same(&filter.get_state(), &before);
        // A valid input cannot conceal an SGP4 failure in a model callback.
        filter.tle.eccen = 1.1;
        assert!(filter.update(before.epoch_unix_s + 10.0, 0.0, 1.0).is_err());
        assert_same(&filter.get_state(), &before);
    }
}

#[test]
fn invalid_sigma_point_aborts_whole_update() {
    for kind in ["ukf", "srukf"] {
        let mut filter = filter(kind, State::zeros(), Some(6.63));
        let mut covariance = filter.backend.covariance();
        covariance[(2, 2)] = 400e6_f64.powi(2);
        filter.backend = Backend::new(kind, State::zeros(), covariance).unwrap();
        let before = filter.get_state();
        let error = filter
            .update(before.epoch_unix_s + 1.0, 0.0, 1.0)
            .unwrap_err();
        assert!(error.to_string().contains("center frequency"));
        assert_same(&filter.get_state(), &before);
    }
}

#[test]
fn covariance_validation_rejects_asymmetry_and_nonpositive_matrices() {
    for covariance in [
        Covariance::zeros(),
        Covariance::new([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        Covariance::new([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ] {
        assert!(validate_covariance(&covariance).is_err());
    }
}
