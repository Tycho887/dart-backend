"""The final-gate timeline uses the same 15 contacts as the DOCX tables."""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "generate_final_gate_timeline.py"
SPEC = importlib.util.spec_from_file_location("final_gate_timeline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
timeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = timeline
SPEC.loader.exec_module(timeline)


def test_final_gate_timeline_manifest_and_generated_png(tmp_path: Path):
    valid = timeline.load_valid_contact_ids(ROOT / timeline.DEFAULT_MANIFEST)
    assert len(valid) == 15
    output = tmp_path / "timeline.png"
    timeline.generate(
        ROOT / timeline.DEFAULT_CONTACTS,
        ROOT / timeline.DEFAULT_MANIFEST,
        output,
    )
    payload = output.read_bytes()
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", payload[16:24])
    assert width >= 2500
    assert height >= 1000
    assert b"inclusive final gates" in payload
    assert b"at least 250 accepted Doppler measurements" in payload
