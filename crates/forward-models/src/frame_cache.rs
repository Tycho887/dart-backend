//! Bounded memoization of exact satkit TEME→GCRF rotations.
//!
//! Epoch keys are satkit's native continuous microseconds, never rounded UTC.
//! Runtime Earth-orientation data must stay fixed during a solve/evaluation.
use crate::{FmResult, ForwardModelError};
use lru::LruCache;
use numeris::Quaternion;
use satkit::frametransform::rotation;
use satkit::{Frame, Instant};
use std::num::NonZeroUsize;
use std::sync::{Mutex, OnceLock};

type Rotations = LruCache<i64, Quaternion<f64>>;
static ROTATIONS: OnceLock<Mutex<Rotations>> = OnceLock::new();

pub fn teme_to_gcrf(time: &Instant) -> FmResult<Quaternion<f64>> {
    let cache =
        ROTATIONS.get_or_init(|| Mutex::new(LruCache::new(NonZeroUsize::new(1_048_576).unwrap())));
    if let Some(value) = cache.lock().unwrap().get(&time.raw) {
        return Ok(*value);
    }
    let value = rotation(Frame::TEME, Frame::GCRF, time)
        .map_err(|e| ForwardModelError::Propagation(e.to_string()))?;
    cache.lock().unwrap().put(time.raw, value);
    Ok(value)
}

/// Call after replacing satkit Earth-orientation data in a long-running process.
pub fn clear() {
    if let Some(cache) = ROTATIONS.get() {
        cache.lock().unwrap().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numeris::Vector3;
    use satkit::frametransform::transform_state;
    #[test]
    fn cached_rotations_match_state_dispatch_at_distinct_subsecond_epochs() {
        let r = Vector3::from_array([7e6, -1e5, 2e5]);
        let v = Vector3::from_array([100.0, 7500.0, -200.0]);
        for delta in [-1.0, -0.707, 0.0, 0.350, 0.350001, 1.0] {
            let t = Instant::from_unixtime(1777852800.0 + delta);
            let (expected_r, expected_v) =
                transform_state(Frame::TEME, Frame::GCRF, &t, &r, &v).unwrap();
            for _ in 0..2 {
                let q = teme_to_gcrf(&t).unwrap();
                assert!((q * r - expected_r).norm() < 1e-9);
                assert!((q * v - expected_v).norm() < 1e-12);
            }
        }
        clear();
    }
}
