# AGENT SPECIFICATION: Orbit Estimation and Sensor Sensitivity Engine

This document specifies the requirements, mathematical formulations, and validation steps for completing the unified orbit estimation and sensitivity engine in Rust.

---

## 1. System Overview and Scope

The engine provides a dual-mode API for orbit determination and trajectory analysis:

1. **Sequential/Step-wise Interface**: Provides instantaneous states, step-to-step state transition matrices (STMs) $\Phi(t_k, t_{k-1})$, and local measurement sensitivity matrices $H_k$ for sequential estimation (Extended Kalman Filtering, Unscented Kalman Filtering, RTS Smoothing) and trajectory optimization/control (Model Predictive Control).
2. **Batch Interface**: Assembles concatenated residual vectors and multi-time epoch Jacobians $J \in \mathbb{R}^{N \times M}$ mapped to epoch $t_0$ for non-linear least-squares solvers (such as SciPy `least_squares`).

### Operational Scope

* **Target Layout**: Entire implementation consolidated within a single file (`src/lib.rs` or `src/estimation.rs`).
* **Active Measurement Model**: **Doppler shift** only (with pass-specific bias estimation).
* **Deferred Models**: `TrueRange`, `PseudorangePhase`, and `PseudorangeCode` must remain marked with `todo!()`.
* **Execution Sequence**: Complete **Phase 1 (Rust Unit Tests)** and ensure compilation before filling model bodies.

---

## 2. Mathematical Formulation

### 2.1 State Vector and Coordinate Frame

The state vector in the Geocentric Celestial Reference Frame (GCRF) is defined as:


$$x(t) = \begin{bmatrix} r(t) \\ v(t) \end{bmatrix} \in \mathbb{R}^6$$

For batch estimation involving $P$ ground station passes, the augmented parameter vector is:


$$X = \begin{bmatrix} x(t_0) \\ b_{\text{pass}, 1} \\ \vdots \\ b_{\text{pass}, P} \end{bmatrix} \in \mathbb{R}^{6 + P}$$

### 2.2 Relative Geometry and Range Rate

Given ground station position $r_{\text{stn}}$ and velocity $v_{\text{stn}}$ in GCRF:


$$\Delta r = r(t) - r_{\text{stn}}(t), \quad \rho = \Vert{}\Delta r\Vert{}$$

$$\Delta v = v(t) - v_{\text{stn}}(t)$$

$$\dot{\rho} = \frac{\Delta r \cdot \Delta v}{\rho}$$

### 2.3 Doppler Measurement Model

The received Doppler frequency shift $f_d$ for center frequency $f_c$ with carrier speed $c$ and pass bias $b_p$ is:


$$h(x(t), b_p) = -\frac{f_c}{c} \dot{\rho} + b_p$$

The instantaneous sensitivity with respect to Cartesian state $x(t) = [r^T, v^T]^T$ is:


$$\frac{\partial h}{\partial x} = -\frac{f_c}{c} \begin{bmatrix} \frac{\partial \dot{\rho}}{\partial r} & \frac{\partial \dot{\rho}}{\partial v} \end{bmatrix}$$


where:


$$\frac{\partial \dot{\rho}}{\partial r} = \frac{\Delta v^T - \dot{\rho} \frac{\Delta r^T}{\rho}}{\rho}, \quad \frac{\partial \dot{\rho}}{\partial v} = \frac{\Delta r^T}{\rho}$$

### 2.4 State Transition and Batch Sensitivity Chain Rule

For an observation at time $t_k$ corresponding to pass index $p$:


$$\frac{\partial h_k}{\partial x(t_0)} = \frac{\partial h_k}{\partial x(t_k)} \Phi(t_k, t_0)$$

$$\frac{\partial h_k}{\partial b_j} = \begin{cases} 1.0, & \text{if } j = p \\ 0.0, & \text{otherwise} \end{cases}$$

---

## 3. Implementation Plan

### Phase 1: Test-Driven Development (TDD)

Before completing the production algorithms, write the Rust test module (`#[cfg(test)]`) at the bottom of the file. The tests must validate:

1. **Range-Rate Jacobian Consistency**: Compare analytical `compute_range_rate_jacobian` against two-sided central finite differences over perturbed satellite position and velocity components ($h = 1.0\text{ m}$, $h = 10^{-3}\text{ m/s}$). Relative error must be $< 10^{-6}$.
2. **Doppler Sign and Magnitude**: Verify that positive range-rate (receding satellite) produces a negative Doppler shift frequency offset.
3. **Chain Rule Verification**: Ensure that multiplying a known $1 \times 6$ sensitivity vector by a known $6 \times 6$ identity or scaled transition matrix produces the exact projected row.
4. **Dimension Assertions on Batch Evaluation**: Verify that $N$ measurements and $P$ pass parameters generate a residual vector of size $N$ and a Jacobian matrix of size $N \times (6 + P)$.
5. **Deferred Measurement Panics**: Ensure that calling `evaluate_local_sensor` with `MeasurementKind::TrueRange`, `PseudorangePhase`, or `PseudorangeCode` panics with an explicit `todo!()` message.

### Phase 2: Core Data Structures

Retain and expose the following structures with zero-copy-ready layouts:

```rust
use numeris::{DynMatrix, DynVector, Matrix};
use satkit::Instant;

#[repr(C)]
#[derive(Clone, Debug)]
pub struct TrajectoryStep {
    pub time: Instant,
    pub state: DynVector<f64>,       // Length 6: [x, y, z, vx, vy, vz]
    pub phi_step: DynMatrix<f64>,    // 6x6: Phi(t_k, t_{k-1})
    pub phi_epoch: DynMatrix<f64>,   // 6x6: Phi(t_k, t_0)
}

#[derive(Clone, Debug)]
pub struct TrajectoryArc {
    pub epoch: Instant,
    pub steps: Vec<TrajectoryStep>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeasurementKind {
    Doppler,
    TrueRange,
    PseudorangePhase,
    PseudorangeCode,
}

#[derive(Clone, Debug)]
pub struct ObservationRecord {
    pub time: Instant,
    pub kind: MeasurementKind,
    pub observed: DynVector<f64>,
    pub noise_cov: DynMatrix<f64>,
    pub receiver_id: u32,
    pub pass_index: usize,
}

#[derive(Clone, Debug)]
pub struct StepObservationEval {
    pub time: Instant,
    pub residual: DynVector<f64>,
    pub predicted: DynVector<f64>,
    pub h_state: DynMatrix<f64>,
    pub h_params: Option<DynMatrix<f64>>,
}

#[derive(Clone, Debug)]
pub struct BatchEvaluationResult {
    pub residuals: DynVector<f64>,
    pub jacobian: DynMatrix<f64>,
    pub weights: Option<DynVector<f64>>,
}

```

### Phase 3: Algorithmic Completion Details

1. **`TrajectoryArc::evaluate_at`**:
* Implement binary search over `steps` by timestamp.
* If exact match: return cloned state and `phi_epoch`.
* If between steps: implement Hermite cubic or linear interpolation for state; linear interpolation for `phi_epoch`.


2. **`EstimationEngine::evaluate_local_sensor`**:
* Inspect `obs.kind`.
* If `MeasurementKind::Doppler`:
* Compute station state in GCRF via `station_gcrf_state`.
* Compute relative position, relative velocity, and range rate.
* Calculate predicted Doppler using center frequency and carrier speed.
* Calculate residual: $r_k = \text{observed} - \text{predicted}$.
* Compute analytical measurement Jacobian $H_{\text{state}} = -\frac{f_c}{c} H_{\text{range\_rate}}$.
* Return `StepObservationEval` with `h_params` populated as a $1 \times P$ matrix containing $1.0$ at column `obs.pass_index`.


* If `MeasurementKind::TrueRange`, `PseudorangePhase`, or `PseudorangeCode`:
* Invoke `todo!("MeasurementKind not yet implemented")`.




3. **`EstimationEngine::batch_evaluate`**:
* Assemble flat `DynVector` and `DynMatrix` without allocating intermediate row vectors.
* Pre-allocate total rows $R = \sum_i \dim(y_i)$ and columns $C = 6 + P$.
* Chain rule: $J_{\text{state}} = H_{\text{state}} \cdot \Phi(t_k, t_0)$.
* Write directly into slice views or matrix entries.


4. **SGP4 Finite Differencing Engine**:
* Complete `compute_sgp4_stm` with parameter-scaled perturbations ($10^{-6}$ relative step).
* Support parameter sets for semi-major axis / mean motion, eccentricity, and inclination.



---

## 4. Phase 1 Rust Unit Tests Template

The agent must implement the following tests in the single-file codebase under `#[cfg(test)]`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    const SPEED_OF_LIGHT: f64 = 299_792_458.0;

    fn generate_mock_vectors() -> (Matrix<f64, 3, 1>, Matrix<f64, 3, 1>, Matrix<f64, 3, 1>, Matrix<f64, 3, 1>) {
        let pos_stn = Matrix::new([[6378137.0], [0.0], [0.0]]);
        let vel_stn = Matrix::new([[0.0], [465.1], [0.0]]);
        let pos_sat = Matrix::new([[7000000.0], [1000000.0], [2000000.0]]);
        let vel_sat = Matrix::new([[500.0], [7200.0], [1500.0]]);
        (pos_stn, vel_stn, pos_sat, vel_sat)
    }

    #[test]
    fn test_range_rate_jacobian_against_finite_differences() {
        let (pos_stn, vel_stn, pos_sat, vel_sat) = generate_mock_vectors();
        let h_analytical = compute_range_rate_jacobian(&pos_stn, &vel_stn, &pos_sat, &vel_sat);

        let eps_pos = 1.0;     // 1 meter perturbation
        let eps_vel = 1e-3;    // 1 mm/s perturbation

        let mut h_fd = Matrix::<f64, 1, 6>::zeros();

        // Finite differences for position components
        for i in 0..3 {
            let mut p_plus = pos_sat;
            let mut p_minus = pos_sat;
            p_plus[i] += eps_pos;
            p_minus[i] -= eps_pos;

            let rr_plus = range_rate(&pos_stn, &vel_stn, &p_plus, &vel_sat);
            let rr_minus = range_rate(&pos_stn, &vel_stn, &p_minus, &vel_sat);
            h_fd[(0, i)] = (rr_plus - rr_minus) / (2.0 * eps_pos);
        }

        // Finite differences for velocity components
        for i in 0..3 {
            let mut v_plus = vel_sat;
            let mut v_minus = vel_sat;
            v_plus[i] += eps_vel;
            v_minus[i] -= eps_vel;

            let rr_plus = range_rate(&pos_stn, &vel_stn, &pos_sat, &v_plus);
            let rr_minus = range_rate(&pos_stn, &vel_stn, &pos_sat, &v_minus);
            h_fd[(0, i + 3)] = (rr_plus - rr_minus) / (2.0 * eps_vel);
        }

        for i in 0..6 {
            let diff = (h_analytical[(0, i)] - h_fd[(0, i)]).abs();
            let denom = h_analytical[(0, i)].abs().max(1e-9);
            assert!(
                diff / denom < 1e-5,
                "Jacobian mismatch at index {}: analytical={}, numerical={}",
                i,
                h_analytical[(0, i)],
                h_fd[(0, i)]
            );
        }
    }

    #[test]
    fn test_doppler_shift_sign() {
        let center_freq = 400.0e6;
        let positive_rr = 1000.0; // Receding
        let negative_rr = -1000.0; // Approaching

        let d_pos = doppler_shift(positive_rr, center_freq);
        let d_neg = doppler_shift(negative_rr, center_freq);

        assert!(d_pos < 0.0, "Receding object must induce negative Doppler shift");
        assert!(d_neg > 0.0, "Approaching object must induce positive Doppler shift");
    }

    #[test]
    #[should_panic(expected = "not yet implemented")]
    fn test_deferred_measurement_panics() {
        let engine = EstimationEngine {
            force_model: satkit::orbitprop::ForceModel::default(),
        };
        let dummy_state = DynVector::<f64>::zeros(6);
        let dummy_obs = ObservationRecord {
            time: Instant::now(),
            kind: MeasurementKind::TrueRange,
            observed: DynVector::<f64>::zeros(1),
            noise_cov: DynMatrix::<f64>::zeros(1, 1),
            receiver_id: 1,
            pass_index: 0,
        };

        let _ = engine.evaluate_local_sensor(&dummy_state, &dummy_obs);
    }
}

```

---

## 5. Acceptance Criteria

The implementation is complete when:

1. `cargo test` passes all tests in the embedded test module without warnings.
2. The `compute_range_rate_jacobian` function matches central finite differences to within $10^{-5}$ relative error across non-degenerate orbital geometries.
3. `batch_evaluate` correctly handles multi-pass observations, producing identical results to sequentially evaluating `step_evaluate` and projecting rows via $H_k \Phi(t_k, t_0)$.
4. All non-Doppler measurement branches (`TrueRange`, `PseudorangePhase`, `PseudorangeCode`) fail with clear `todo!()` panic messages.
5. All code remains strictly confined to a single source file without external dependencies beyond `satkit`, `numeris`, and standard library components.