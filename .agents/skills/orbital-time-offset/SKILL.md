---
name: orbital-time-offset
description: Design, implement, review, or test guarded during-pass time-offset control through an antenna's Orbital API.
---

# Orbital time-offset control

Read [references/time-offset-mode.md](references/time-offset-mode.md) before
changing the live controller or transport. The detailed source guidance is
maintained in `skills/orbital-time-offset/`; keep this registered copy aligned.

- Separate estimator, policy, transport, and ADX readback confirmation.
- Use seconds with `_s` names and absolute targets internally.
- Bind each controller to a contact, antenna, ephemeris, profile, and model.
- Default to dry-run. Live writes require explicit immediate authorization and
  confirmed endpoint semantics/sign; never use a fallback antenna identity.
- Permit one unconfirmed write. An HTTP acknowledgement is not proof that the
  value was applied; wait for matching fresh ADX readback.
- Use UKF feed-forward plus slow PI (`Kd = 0`), saturation, anti-windup,
  deadband, rate/slew limits, irregular-dt handling, and bumpless acquisition.
- Reset to zero on pass completion and controlled cancellation. Fault and alert
  if application or reset remains ambiguous.
- A booked shadow contact is distinct from controller dry-run mode.
