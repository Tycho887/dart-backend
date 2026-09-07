---
name: shadow-pass-scheduling
description: Plan, review, test, or execute external shadow-contact bookings with KOGS, including conflicts, idempotency, audit, and compensation.
---

# Shadow-pass scheduling

Read [references/safety.md](references/safety.md) before changing booking code.

- Planning is the default and performs no external writes.
- Require explicit user authorization immediately before booking.
- Load antenna, stations, mission profile, lead time, margins, search window,
  timeout, and audit destination from reviewed deployment configuration.
- Capture one UTC `now`, validate credentials first, use bounded timeouts, and
  reject malformed or incomplete typed responses.
- Select deterministically. Check state, spacecraft and ephemeris identity,
  tracking duration, setup/teardown conflicts, and existing idempotency keys.
- Derive idempotency from source contact, target antenna, and mission profile.
- Persist request/response digests and lifecycle transitions without secrets.
- Treat booking, ephemeris assignment, and persistence as a saga. Cancel after
  downstream failure, or record reconciliation-required status and alert.
- Booking authorization never authorizes Orbital offset writes.
