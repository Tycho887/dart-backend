"""DART usage demo: build an Sgp4Input, run it through the Rust solver, export TDM.

Real telemetry flows through the loaders (``dart.loaders.leo`` / ``dart.loaders.cislunar``,
backed by ADX + KOGS). This demo runs fully offline with synthetic data to show the
transport contract: dataclasses -> msgpack -> Rust -> msgpack -> dataclasses -> TDM.
"""

from dart.tdm.legacy import write_result_tdm, write_tdm
from dart.schema import Observation, Sgp4Input, SolverOptions, Station, Tle
from dart.solver import solve

TLE_LINE_1 = "1 25544U 98067A   24001.00000000  .00016717  00000-0  10270-3 0  9993"
TLE_LINE_2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.50102972239846"


def demo_input() -> Sgp4Input:
    return Sgp4Input(
        spacecraft_id="25544",
        epoch_unix=1_704_067_200.0,
        tle=Tle(line1=TLE_LINE_1, line2=TLE_LINE_2),
        stations=[
            Station(id="sys-1", name="Svalbard", lat_deg=78.23, lon_deg=15.39, alt_km=0.055),
        ],
        observations=[
            Observation(
                epoch_unix=1_704_067_300.0,
                doppler_hz=-1234.5,
                azimuth_deg=182.3,
                elevation_deg=37.2,
                station_id="sys-1",
            ),
        ],
        options=SolverOptions(max_iterations=25),
    )


def main() -> None:
    inp = demo_input()
    result = solve(inp)

    print(f"solve() -> mode={result.mode} success={result.success} message={result.message!r}")

    write_tdm(inp, "input.tdm")
    write_result_tdm(result, "result.tdm")
    print("wrote input.tdm and result.tdm")


if __name__ == "__main__":
    main()
