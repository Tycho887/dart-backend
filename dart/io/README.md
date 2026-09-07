# `dart.io`

`dart.io` is the external-data boundary for DART. Provider modules know one
external interface each; `load.py` is the sole multi-provider orchestration
module; `contact.py` contains shared record types; `measurement.py` owns the
canonical DataFrame contract.

Keep model selection, numerical propagation, optimizer settings, TDM
serialization, and controller policy outside this package. Add provider fields
only when they preserve source data or provenance needed by downstream code.
