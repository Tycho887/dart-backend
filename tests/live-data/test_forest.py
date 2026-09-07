"""Explicitly enabled live experiments; credentials and prior IDs are required."""

import asyncio
import os
import runpy
from pathlib import Path

import pytest
from dotenv import load_dotenv

from dart.io.adx import client_from_env
from dart.io.oem import read_oem
from dart.od import OrbitModel
from experiments.live_data import ExperimentSettings, run_comparison


@pytest.mark.live_data
@pytest.mark.parametrize("name", ["forest16", "forest17", "forest18", "forest19"])
def test_forest(name: str, tmp_path: Path) -> None:
    if os.getenv("DART_RUN_LIVE_DATA") != "1":
        pytest.skip("set DART_RUN_LIVE_DATA=1 to run live orbit experiments")
    load_dotenv(
        os.getenv("DART_SECRETS_ENV", "/opt/dart/secrets/test.env"), override=False
    )
    prefix = f"DART_{name.upper()}"
    required = [f"{prefix}_EPHEMERIS_ID", f"{prefix}_OEM", "KOGS_API_KEY"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        pytest.fail(f"explicit live inputs required: {', '.join(missing)}")
    case = runpy.run_path(str(Path(__file__).with_name(f"{name}.py")))
    reference = read_oem(Path(os.environ[f"{prefix}_OEM"]))
    root = Path(os.environ.get("DART_LIVE_DATA_OUTPUT", str(tmp_path)))
    settings = ExperimentSettings(OrbitModel.SGP4, case["CENTER_FREQUENCY_HZ"])
    with client_from_env() as client:
        results = asyncio.run(
            run_comparison(
                case["CONTACT_IDS"],
                ephemeris_id=os.environ[f"{prefix}_EPHEMERIS_ID"],
                spacecraft_id=case["SPACECRAFT_ID"],
                settings=settings,
                reference=reference,
                output_dir=root / name,
                kogs_api_key=os.environ["KOGS_API_KEY"],
                adx_client=client,
            )
        )
    assert results, "no fit cases were executed"
    assert {result.output.model_kind for result in results} == set(OrbitModel)
    # Accuracy and nonconvergence are reported scientific outcomes, not gates.
    assert (root / name / "summary.json").is_file()
