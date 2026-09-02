# main.py:

What is the purpose of this demo, and why does it seemingly create a TDM file separate from the rest of the TDM logic?
This entire file and dart.tdm does seemingly not fulfill any purpose and may be deleted.

Forward models:

This is extremely poorly implemented. Splitting logic between a rust core and python core is asking for problems with implementation divergence. 
Path forward: Identify specifically what python features are necessary or argue why logic should be kept in python. If no specific feature can be found, move the logic to rust.

# Dataclasses:

Why are there so many dataclasses! I can find 58 instances accross 20 files,
things like "solverOptions", "filteridentity" etc have quite few arguments. Perhaps some of these could be removed and the options moved to function parameters? This is particularly when a dataclass is only used once. 

Some examples of where containing data is very useful is for API requests, as this captures arguments that may or may not be present and will likely be reused multiple times. A clear example where this has gone wrong is SolverOptions, where "ref_frame" seemingly allows for use of GCRF reference frames for SGP4. This is a hallucination, and in reality the SGP4-based solver only uses TEME/GCRF as a intermediate before passing it through the "doppler plant" which defines expected range-rate values.

# Rust:

The rust base is slightly better, but overall can be drastically improved. The use of SLSQP seems to work, however we might also consider the option of defining rust functions such that we only use rust for computing Jacobians/Residuals, and use python-based scipy-optimizers for the actual solving. This can be considered a middle ground to best utilize the well-documented leastSquares package, and could allow for simpler rust code.

In this case, we might consider a minimally viable rust-base. A suggestion could be:

Compute_metrics_time_bias_center(sk.TLE, rotation_cache, timestamps, doppler_values, time_offset, bias, center_freq) -> Jacobian-Matrix + Residual-vector

We then define similar minimal functions for the different modes. Reasonably we could then reuse these rust-based forward models for different parts of the codebase, be it control systems or optimizers- and allow us to test it. The rotation cache is a mandatory element which is precomputed and stores position information for the ground station.

Some important notes: Always use GCRF as intermediates for 3D position. If using SGP4 or a high-fidelity model, rotate into this frame. There is no reason why a different frame should be used, and theres no reason to include it in a struct parameter.

# Rust-Python integration

there are too many layers of abstraction, low coherence and location logic. If I want to work on the optimizer for time-models, what is the entry-point? Why are 10 dataclasses defined in different files used willy-nilly.

Inter-module imports. At the current time, we cannot state that the "API" section of DART or the "solver" section is completed, as the different parts of the code import each other. For instance, in an operational codebase we'd often need to interact with kogs or ADX/Kusto/book shadow passes/control APIs, currently all of this logic is spread accross scripts in different locations.
Ideally dart.io should expose all dataclasses and functions necessary to use these APIs and simplify everything else. dart.tdm should handle all tdm logic, dart.od should handle all orbit-determination logic, and dart-control should define real-time algorithms.
The control code should not be dependent on dart.io, and not share any dataclasses if possible. 
Instead we define logic-orchestrators which import dart.io, dart.od etc, use the functions defined here, and process the data. This way we can easily test each submodule independently and work on features without breaking other logic.
dart.models can be the rust code, and defines the forward models which can be used later in either optimizers or controllers.
service/ seems to be a bag of random code again, so consider refactoring/moving some of this code.

# Excessive duplication and code

A dart prototype was 15 files and maybe 2000 lines of code and had more features than the current version. We can reasonably have /scripts, /tests etc. but these should be consistent in what they contain. forest-experiment and write-tdm have very different meanings. Development/testing code, can be moved to dev.

