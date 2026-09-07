# SIGMET product

Sources: `TDM-SignalMetrics _ Ranging.pdf`, pp. 1–2; `Appendix-A3 Example Signal
Metrics Data TDM File _ Ranging.pdf`, p. 1.

Additional metadata names the `TRANSMIT_BAND` and `RECEIVE_BAND`, each `S`, `X`,
or `Ka`.

| Keyword | Units | Resolution | Meaning |
| --- | --- | --- | --- |
| `CARRIER_POWER` | dBW | 0.01 | Total received signal power |
| `PC_N0` | dB-Hz | 0.01 | Total received signal power / noise density |
| `PR_N0` | dB-Hz | 0.01 | Ranging-component power / noise density |

Data timestamps have one-second resolution. The example says measurements are
usually produced around once per second, but that is not a mandatory cadence.

Do not substitute Eb/N0 or Es/N0 merely because ADX exposes those columns.
Require an explicit, site-confirmed mapping to the stated KSAT metric.
