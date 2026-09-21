//! Line-oriented benchmark worker. All I/O and serialization are outside timing.

use _forward_models::{
    BatchEvaluationResult, EstimationEngine, MeasurementKind, ObservationRecord,
    filters::{Covariance, DopplerFilter, FilterState, State},
    hifi_evaluate, hifi_evaluate_augmented, lofi_evaluate, lofi_evaluate_augmented,
    lofi_evaluate_epoch, propagate_sgp4_gcrf, propagate_states, tle_with_offset,
};
use numeris::{DynMatrix, DynVector, Vector6};
use satkit::{ITRFCoord, Instant, TLE, orbitprop::PropSettings};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    error::Error,
    hint::black_box,
    io::{self, BufRead, Write},
    time::Instant as Clock,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Deserialize)]
struct Fixture {
    tle: [String; 2],
    receiver: [f64; 3],
    frequency_hz: f64,
    epoch: f64,
    epochs: Vec<f64>,
    observed_hz: Vec<f64>,
    variance_hz2: f64,
    nominal: [f64; 6],
    initial_state: [f64; 3],
    initial_covariance: [[f64; 3]; 3],
    process_noise_rates: [f64; 3],
}

struct Inputs {
    fixture: Fixture,
    tle: TLE,
    engine: EstimationEngine,
    observations: Vec<ObservationRecord>,
    times: Vec<Instant>,
    epoch: Instant,
    nominal: Vector6<f64>,
    settings: PropSettings,
}

impl Inputs {
    fn new(fixture: Fixture) -> Result<Self> {
        if fixture.epochs.is_empty() || fixture.epochs.len() != fixture.observed_hz.len() {
            return Err("fixture must contain matching, nonempty epochs and observations".into());
        }
        let tle = TLE::load_2line(&fixture.tle[0], &fixture.tle[1])?;
        let [lat, lon, alt] = fixture.receiver;
        let engine = EstimationEngine {
            receivers: vec![ITRFCoord::from_geodetic_deg(lat, lon, alt)],
            center_frequency: fixture.frequency_hz,
            num_passes: 1,
        };
        let times: Vec<_> = fixture
            .epochs
            .iter()
            .map(|t| Instant::from_unixtime(*t))
            .collect();
        let observations = times
            .iter()
            .zip(&fixture.observed_hz)
            .map(|(time, value)| ObservationRecord {
                time: *time,
                kind: MeasurementKind::Doppler,
                observed: DynVector::from_vec(vec![*value]),
                noise_cov: DynMatrix::from_rows(1, 1, &[fixture.variance_hz2]),
                receiver_id: 0,
                pass_index: 0,
            })
            .collect();
        Ok(Self {
            tle,
            engine,
            observations,
            times,
            epoch: Instant::from_unixtime(fixture.epoch),
            nominal: Vector6::from_array(fixture.nominal),
            settings: PropSettings::default(),
            fixture,
        })
    }

    fn filter(&self, kind: &str) -> Result<DopplerFilter> {
        let f = &self.fixture;
        let [lat, lon, alt] = f.receiver;
        Ok(DopplerFilter::new(
            self.tle.clone(),
            ITRFCoord::from_geodetic_deg(lat, lon, alt),
            f.frequency_hz,
            f.epoch,
            State::from_array(f.initial_state),
            Covariance::new(f.initial_covariance),
            State::from_array(f.process_noise_rates),
            kind,
            None,
        )?)
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Family {
    Batch,
    Trajectory,
    Filter,
}

#[derive(Deserialize)]
struct Case {
    family: Family,
    mode: String,
    x: Vec<f64>,
    drag: f64,
    filter_kind: String,
}

#[derive(Deserialize)]
struct Request {
    case: Case,
    validate: bool,
}

// Keep native result types until AFTER the clock stops.
enum Output {
    Batch(BatchEvaluationResult),
    Sgp4States(Vec<Vector6<f64>>),
    FullStates(Vec<DynVector<f64>>),
    Constructed(DopplerFilter),
    Snapshot(FilterState),
    Updated(usize, Option<f64>),
}

fn batch(case: &Case, i: &Inputs) -> Result<Output> {
    let result = match case.mode.as_str() {
        "sgp4" => lofi_evaluate(&i.engine, &case.x, &i.tle, &i.observations),
        "sgp4_augmented" => lofi_evaluate_augmented(&i.engine, &case.x, &i.tle, &i.observations),
        "sgp4_epoch" => lofi_evaluate_epoch(&i.engine, &case.x, &i.tle, &i.observations),
        "full_state" => hifi_evaluate(
            &i.engine,
            &case.x,
            &i.nominal,
            &i.epoch,
            &i.observations,
            &i.settings,
            case.drag,
        ),
        "full_state_augmented" => hifi_evaluate_augmented(
            &i.engine,
            &case.x,
            &i.nominal,
            &i.epoch,
            &i.observations,
            &i.settings,
            case.drag,
        ),
        _ => return Err(format!("unknown batch mode: {}", case.mode).into()),
    }?;
    Ok(Output::Batch(result))
}

fn trajectory(case: &Case, i: &Inputs) -> Result<Output> {
    match case.mode.as_str() {
        "sgp4" => {
            let tle = tle_with_offset(&i.tle, &case.x)?;
            Ok(Output::Sgp4States(propagate_sgp4_gcrf(&tle, &i.times)?))
        }
        "full_state" => {
            let states = propagate_states(&i.nominal, &i.epoch, &i.times, &i.settings, case.drag)?;
            Ok(Output::FullStates(states))
        }
        _ => Err(format!("unknown trajectory mode: {}", case.mode).into()),
    }
}

fn filter_operation(case: &Case, i: &Inputs, filter: &mut DopplerFilter) -> Result<Output> {
    match case.mode.as_str() {
        "construct" => Ok(Output::Constructed(i.filter(&case.filter_kind)?)),
        "predict" => {
            let mut state = filter.predict(i.fixture.epochs[0])?;
            for epoch in &i.fixture.epochs[1..] {
                state = black_box(filter.predict(*epoch)?);
            }
            Ok(Output::Snapshot(state))
        }
        "update" => {
            let mut accepted = 0;
            let mut nis = None;
            for (epoch, value) in i.fixture.epochs.iter().zip(&i.fixture.observed_hz) {
                nis = black_box(filter.update(*epoch, *value, i.fixture.variance_hz2)?);
                accepted += usize::from(nis.is_some());
            }
            Ok(Output::Updated(accepted, nis))
        }
        "snapshot" => {
            let mut state = black_box(&*filter).get_state();
            for _ in 1..i.fixture.epochs.len() {
                state = black_box(black_box(&*filter).get_state());
            }
            Ok(Output::Snapshot(state))
        }
        _ => Err(format!("unknown filter mode: {}", case.mode).into()),
    }
}

fn snapshot(state: FilterState) -> Value {
    let covariance: Vec<Vec<f64>> = (0..3)
        .map(|r| (0..3).map(|c| state.covariance[(r, c)]).collect())
        .collect();
    json!({"epoch": state.epoch_unix_s, "state": state.state.as_slice(), "covariance": covariance})
}

fn reference(output: Output, filter: &DopplerFilter) -> Value {
    match output {
        Output::Batch(result) => {
            let j = &result.residual_jacobian;
            let rows: Vec<Vec<f64>> = (0..j.nrows())
                .map(|r| (0..j.ncols()).map(|c| j[(r, c)]).collect())
                .collect();
            json!({"residuals": result.residuals.as_slice(), "jacobian": rows})
        }
        Output::Sgp4States(states) => {
            json!({"states": states.iter().map(|v| v.as_slice()).collect::<Vec<_>>()})
        }
        Output::FullStates(states) => {
            json!({"states": states.iter().map(|v| v.as_slice()).collect::<Vec<_>>()})
        }
        Output::Constructed(f) => snapshot(f.get_state()),
        Output::Snapshot(state) => snapshot(state),
        Output::Updated(accepted, nis) => {
            let mut value = snapshot(filter.get_state());
            value["accepted"] = json!(accepted);
            value["nis"] = json!(nis);
            value
        }
    }
}

fn run(request: &Request, inputs: &Inputs) -> Result<Value> {
    // Reset outside timing, including for warm-ups and validation.
    let mut filter = inputs.filter(&request.case.filter_kind)?;
    let started = Clock::now();
    let output = black_box(match request.case.family {
        Family::Batch => batch(&request.case, inputs)?,
        Family::Trajectory => trajectory(&request.case, inputs)?,
        Family::Filter => filter_operation(&request.case, inputs, &mut filter)?,
    });
    let elapsed_ns = started.elapsed().as_nanos();
    let mut response = json!({"elapsed_ns": elapsed_ns});
    if request.validate {
        response["reference"] = reference(output, &filter);
    }
    Ok(response)
}

fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err("benchmark requires cargo build --release".into());
    }
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let fixture: Fixture = serde_json::from_str(&lines.next().ok_or("missing fixture")??)?;
    let inputs = Inputs::new(fixture)?;
    let mut stdout = io::stdout().lock();
    writeln!(stdout, "{}", json!({"ready": true, "profile": "release"}))?;
    stdout.flush()?;
    for line in lines {
        let request: Request = serde_json::from_str(&line?)?;
        writeln!(stdout, "{}", run(&request, &inputs)?)?;
        stdout.flush()?;
    }
    Ok(())
}
