import csv
from pathlib import Path

import numpy as np
import pytest
import satkit

from dart.gmat import tai_mjd, utc, utc_mjd
from dart.loaders.gps import holdout_mask, load_bestxyz, receiver_epoch


def test_receiver_epoch_uses_embedded_time_and_resolves_week_rollover():
    # GPS week 2417 starts 18 seconds before Sunday UTC midnight.
    t = satkit.time.from_gps_week_and_second(2417, 0.1).as_unixtime()
    assert receiver_epoch(round((t + 0.4) * 1000), 100) == pytest.approx(t, abs=1e-6)
    t = satkit.time.from_gps_week_and_second(2416, 604799.9).as_unixtime()
    assert receiver_epoch(round((t + 1) * 1000), 604799900) == pytest.approx(t, abs=1e-6)
    with pytest.raises(ValueError, match="seconds of week"):
        receiver_epoch(round(t * 1000), 604800000)


def test_gmat_time_origin_and_tai_utc_offset():
    t = utc("2026-05-03T12:00:00Z")
    assert utc_mjd(t) == 31164.0
    assert (tai_mjd(t) - utc_mjd(t)) * 86400 == pytest.approx(37, abs=1e-6)


def write_tsv(path: Path, names, rows):
    with path.open("w", encoding="utf-16", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(names)
        writer.writerows(rows)


def test_loader_named_axes_units_duplicates_rejections_missing_velocity(tmp_path):
    epoch = satkit.time.from_gps_week_and_second(2417, 45980.1).as_unixtime()
    packets = [round((epoch + delta) * 1000) for delta in (0.3, 0.4, 60.3, 120.3, 180.3)]
    # Deliberately shuffle columns and provide two packets for the same epoch.
    names = ["Time", "h.bestxyz_z_pos_f.3", "h.bestxyz_y_pos_dev.5", "h.bestxyz_x_pos_f.1",
             "h.bestxyz_z_pos_dev.6", "h.bestxyz_y_pos_f.2", "h.bestxyz_x_pos_dev.4"]
    write_tsv(tmp_path / "FOREST-16-BESTXYZ-position.csv", names, [
        [packets[0], 100, 20, 6900000, 20, 200, 20],
        [packets[1], 100, 2, 6900001, 3, 200, 1],
        [packets[2], 101, 2, 6900002, 3, 201, 1],
        [packets[3], 0, 2, 0, 3, 0, 1],
        [packets[4], 0, 2, 6900000, -1, 0, 1],
    ])
    write_tsv(tmp_path / "FOREST-16-BESTXYZ-time.csv", ["Time", "h.bestxyz_gps_ref_ms.1"], [
        [packet, 45980100 + step * 1000] for packet, step in zip(packets, (0, 0, 60, 120, 180))
    ])
    d = load_bestxyz(tmp_path, "FOREST-16", epoch - 1, epoch + 200)
    assert len(d) == 2
    np.testing.assert_allclose(d.position[0], [6900.001, 0.2, 0.1])
    np.testing.assert_allclose(d.sigma[0], [0.001, 0.002, 0.003])
    np.testing.assert_allclose(d.epoch, [epoch, epoch + 60], rtol=0, atol=1e-6)
    assert np.isnan(d.velocity).all()
    assert {r["reason"] for r in d.rejected} == {"duplicate_epoch", "orbital_radius", "position_uncertainty"}


def test_holdout_selection_has_no_residual_dependency():
    times = np.array([utc(f"2026-05-04T{h}:00Z") for h in ("12:49", "12:50", "12:59", "13:00")])
    assert holdout_mask(times).tolist() == [False, True, True, False]
