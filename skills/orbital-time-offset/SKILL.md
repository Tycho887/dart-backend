---
name: orbital-time-offset
description: Design, implement, review, or test automated during-pass time/epoch-offset control through this repository's Orbital OACP API. Use for the live measurement-to-optimizer-to-antenna loop, its safety gates, pass lifecycle, and recovery; not for offline TLE generation alone.
---

# Orbital time-offset control

Build time-offset mode as a guarded closed loop:

`Doppler samples -> processed measurements -> offset estimate -> acceptance policy -> offset write`

Read [references/time-offset-mode.md](references/time-offset-mode.md) before changing the live controller or Orbital API client. Reinspect the current branch as well: this repository has divergent historical implementations, and the reference distinguishes observed contracts from contracts that still need confirmation.

## Required behavior

- Keep estimation, control policy, and Orbital transport separate. The optimizer proposes an offset; only the controller decides whether it is safe to apply; only the API client performs the write.
- Treat seconds as the internal unit and use the repository-wide `_s` suffix,
  such as `estimated_offset_s` and `applied_offset_s`.
- Bind one controller instance to one contact and one antenna. Reject updates outside that contact's active pass window and prevent concurrent writers for the same antenna.
- Start in dry-run mode. Live writes require explicit configuration and resolved antenna identity; never retain the current hard-coded `SG127` fallback.
- Before enabling live writes, confirm the endpoint's setpoint-versus-increment semantics and sign convention against authoritative Orbital documentation or a controlled antenna test. Do not infer either from the local wrapper.
- Gate estimates on measurement count, freshness, finiteness, confidence, fit quality, and pass phase. Apply configurable absolute bounds, maximum change per update, deadband, and minimum update interval.
- Record every proposed, rejected, clamped, attempted, acknowledged, and reset value with contact ID, antenna, timestamps, estimator diagnostics, and reason codes.
- Update local applied state only after a successful Orbital acknowledgement. On ambiguous failures, enter a fault/hold state instead of assuming the command took effect or blindly retrying.
- Keep requested, API-acknowledged, and ADX-confirmed applied offsets distinct.
  Allow only one write awaiting ADX confirmation; do not integrate PI error or
  retry while application is ambiguous.
- Return the offset to `0` at normal pass completion and on controlled cancellation. Make reset bounded and observable; if it cannot be acknowledged, raise an operational alert rather than concealing the residual offset.

## Implementation workflow

1. Inspect the files named in the reference and reconcile their current types and method names before adding behavior. Do not build on stale interfaces.
2. Capture the confirmed Orbital contract in the client and its tests: URL construction, body encoding, response shape, timeout, and error handling.
3. Implement a pass-scoped state machine: `PLANNED -> PRECHECK -> WAITING ->
   ACQUIRING -> TRACKING -> RESETTING -> COMPLETE`, with `HOLD`, `REACQUIRING`,
   `ABORTED`, or `FAULT` reachable from unsafe estimates and API failures.
4. Make the controller consume an immutable estimate result and return a structured decision. A rejected decision must not call the API.
5. Drive updates from processed-measurement progress or a monotonic cadence, not `counter % N == 0`; the latter fires repeatedly when a counter remains zero or unchanged.
6. Put cleanup around the whole live pass lifecycle, not merely the happy-path loop. Handle shutdown signals and exceptions explicitly.
7. Add unit and integration tests before any live trial, then use a staged rollout: mocked transport, fake Orbital server, dry-run against real pass data, bounded shadow operation, and finally an attended live pass.

Binary-lock acquisition must wait for ADX application confirmation and a
settling interval before scoring consecutive fresh samples. It must find both
lock-window edges, reject an edge touching a sweep bound, confirm the midpoint
remains locked, retain every probe, obey a pass deadline, and reset to zero.

After acquisition, use UKF feed-forward with a deliberately slow PI controller
(`Kd = 0`) until measured-delay simulations and attended tests justify any
derivative action. Include saturation, anti-windup, deadband, slew/rate limits,
irregular cadence handling, and bumpless transfer. The error is the UKF
absolute target minus the ADX-confirmed applied offset.

A booked shadow contact and controller dry-run mode are separate controls. A
booking reserves an external resource; dry-run suppresses Orbital writes.
Require explicit user authorization immediately before either booking or
enabling live writes.

## Completion criteria

Do not call time-offset mode ready until tests demonstrate rejected estimates cause no writes, accepted values obey all bounds and cadence rules, failures do not corrupt applied state, pass ownership prevents conflicting writers, and every terminal path attempts an observable zero reset. Keep live-network tests opt-in and separate from the ordinary test suite.
