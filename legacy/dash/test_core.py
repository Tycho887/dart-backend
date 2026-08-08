"""Fast, offline tests for payload parsing and residual metrics."""

import os
import unittest

import numpy as np

os.environ["DB_LOGGING_ENABLED"] = "false"

from lib.load import TrackingContext
from lib.azure import _kql_datetime, _kql_string
from lib.utils import Result, _iso_to_unix, _safe_bool
from processor import compute_metrics


class PayloadParsingTests(unittest.TestCase):
    def test_boolean_strings_are_parsed_explicitly(self):
        self.assertTrue(_safe_bool("true"))
        self.assertFalse(_safe_bool("false"))

    def test_zero_threshold_is_preserved(self):
        context = TrackingContext.from_payload(
            {"contact_UUID_List": ["a", "b"], "minElevation": 0}
        )

        self.assertEqual(context.contact_uuid_list, "a,b")
        self.assertEqual(context.min_elevation, 0.0)

    def test_iso_timestamp_conversion(self):
        self.assertEqual(_iso_to_unix("1970-01-01T00:00:01Z"), 1.0)
        self.assertIsNone(_iso_to_unix("not-a-timestamp"))

    def test_kql_values_are_validated_and_escaped(self):
        self.assertEqual(_kql_string("sat'id"), "sat''id")
        self.assertEqual(
            _kql_datetime("2026-08-07T10:00:00Z"),
            "2026-08-07T10:00:00+00:00",
        )
        with self.assertRaises(ValueError):
            _kql_datetime("not-a-time")


class MetricTests(unittest.TestCase):
    def test_metrics_for_known_residuals(self):
        result = Result(
            x=np.zeros(3),
            name="demo",
            contact_id="contact-1",
            cov=np.eye(3),
            ssr=2.0,
            residuals=np.array([-1.0, 1.0]),
            success=True,
            message="ok",
            passes_found=1,
        )

        metrics = compute_metrics(result)

        self.assertEqual(metrics["num_samples"], 2)
        self.assertAlmostEqual(metrics["mean"], 0.0)
        self.assertAlmostEqual(metrics["rmse"], 1.0)


if __name__ == "__main__":
    unittest.main()
