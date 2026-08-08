"""Offline tests for the FastAPI endpoint orchestration."""

import os
import unittest
from unittest.mock import Mock, patch

import polars as pl
from fastapi import HTTPException, Request, Response

os.environ["DB_LOGGING_ENABLED"] = "false"

from lib.parseKogs import parse_reservation
from main import _metadata_lookup, receive_spacecraft_data


class TrackingEndpointTests(unittest.TestCase):
    @patch("main.process_telemetry_batch")
    @patch("main.augment_telemetry_dataframe")
    @patch("main.fetch_tracking_data")
    def test_returns_processing_results(self, fetch, augment, process):
        fetch.return_value = pl.DataFrame({"sample": [1]})
        augment.return_value = pl.DataFrame({"spacecraft_id": ["sc-1"]})
        process.return_value = {
            "time_offset_passes": ["contact-1"],
            "mean_elements_windows": [],
            "errors": [],
        }

        response = receive_spacecraft_data({"contact_UUID_List": ["contact-1"]})

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["results"]["time_offset_passes"], ["contact-1"])
        self.assertEqual(process.call_args.kwargs["context"].spacecraft_uuid, "sc-1")

    @patch("main.fetch_tracking_data")
    def test_empty_query_result_is_404(self, fetch):
        fetch.return_value = pl.DataFrame()

        with self.assertRaises(HTTPException) as raised:
            receive_spacecraft_data({"contact_UUID_List": ["contact-1"]})

        self.assertEqual(raised.exception.status_code, 404)


class MetadataEndpointTests(unittest.TestCase):
    def test_logs_proxy_packets_and_returns_request_id(self):
        fetch = Mock(
            return_value=parse_reservation(
                {"id": "contact-1", "state": "READY"}
            )
        )
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/metadata/contact",
                "query_string": b"",
                "headers": [(b"authorization", b"Bearer inbound-secret-marker")],
                "scheme": "http",
                "server": ("localhost", 8000),
                "client": ("127.0.0.1", 12345),
            }
        )
        response = Response()

        with self.assertLogs("db_logger", level="INFO") as captured:
            result = _metadata_lookup(
                fetch,
                "contact-1",
                request,
                response,
                body={"contact_id": "contact-1"},
            )

        request_id = response.headers["X-Request-ID"]
        self.assertEqual(result.id, "contact-1")
        output = "\n".join(captured.output)
        self.assertIn('"event": "proxy_request"', output)
        self.assertIn('"event": "proxy_response"', output)
        self.assertIn(request_id, output)
        self.assertNotIn("inbound-secret-marker", output)
        fetch.assert_called_once_with("contact-1", request_id=request_id)


if __name__ == "__main__":
    unittest.main()
