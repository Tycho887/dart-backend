# Current implementation routing

Use these current files:

- `dart/controller/UKF.py`: identity-bound estimate and diagnostics.
- `dart/controller/acquisition.py`: binary-lock dither and ADX confirmation.
- `dart/controller/controller.py`: PI safety policy and write lifecycle.
- `skills/orbital-time-offset/references/time-offset-mode.md`: full guidance.

Historical `src/api/orbital.py` and `src/misc/ctrl_config.py` files are not
present here and are not authoritative. Confirm transport encoding, sign,
absolute-versus-additive semantics, persistence, and acknowledgement against
current Orbital documentation before enabling any live write.
