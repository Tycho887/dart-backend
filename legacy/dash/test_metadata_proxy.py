"""Offline tests for normalized KOGS metadata proxy functions."""

import os
import json
import unittest
from unittest.mock import patch

import requests

os.environ["DB_LOGGING_ENABLED"] = "false"
os.environ["KOGS_API_KEY"] = "test-only-secret-marker"

from lib.metadata_proxy import (
    MetadataLookupError,
    MetadataNotFound,
    fetch_contact_metadata,
    fetch_ephemeris_metadata,
)


class MetadataProxyTests(unittest.TestCase):
    @staticmethod
    def response(body, status=200, headers=None):
        response = requests.Response()
        response.status_code = status
        response._content = json.dumps(body).encode("utf-8")
        response.headers.update(headers or {"Content-Type": "application/json"})
        response.url = "https://mgmt.kogs.api.ksat.no/24.08/test"
        return response

    @patch("lib.metadata_proxy.get_contact_response")
    def test_contact_response_is_normalized(self, get_contact):
        get_contact.return_value = self.response({
            "contact": {
                "id": "contact-1",
                "state": "READY",
                "spacecraft_id": "spacecraft-1",
                "start_time": "2026-08-07T10:00:00Z",
                "end_time": "2026-08-07T10:10:00Z",
                "ephemeris_id": "ephemeris-1",
            }
        })

        result = fetch_contact_metadata("contact-1", request_id="request-1")

        self.assertEqual(result.id, "contact-1")
        self.assertEqual(result.spacecraft_id, "spacecraft-1")
        self.assertEqual(result.start_time_unix, 1786096800.0)
        self.assertTrue(get_contact.call_args.args[0].startswith("KSAT1-PLAIN "))

    @patch("lib.metadata_proxy.get_ephemeris_response")
    def test_ephemeris_response_is_normalized(self, get_tle):
        get_tle.return_value = self.response({
            "ephemeris_uuid": "ephemeris-1",
            "spacecraft_uuid": "spacecraft-1",
            "kind": "TLE",
            "inline": {"tle": "NAME\nLINE 1\nLINE 2"},
        })

        result = fetch_ephemeris_metadata("ephemeris-1")

        self.assertEqual(result.ephemeris_uuid, "ephemeris-1")
        self.assertEqual(result.inline_tle, "NAME\nLINE 1\nLINE 2")

    def test_identifier_validation_rejects_path_content(self):
        with self.assertRaises(ValueError):
            fetch_contact_metadata("../contact")

    @patch("lib.metadata_proxy.get_contact_response")
    def test_missing_contact_object_is_an_upstream_error(self, get_contact):
        get_contact.return_value = self.response({})
        with self.assertRaises(MetadataLookupError):
            fetch_contact_metadata("contact-1")

    @patch("lib.metadata_proxy.get_contact_response")
    def test_logs_packets_status_and_response_without_credentials(self, get_contact):
        get_contact.return_value = self.response(
            {
                "contact": {"id": "contact-1", "state": "READY"},
                "token": "upstream-token-marker",
            },
            headers={
                "Content-Type": "application/json",
                "Set-Cookie": "cookie-marker",
            },
        )

        with self.assertLogs("db_logger", level="INFO") as captured:
            fetch_contact_metadata("contact-1", request_id="correlation-123")

        output = "\n".join(captured.output)
        self.assertIn('"event": "upstream_request"', output)
        self.assertIn('"event": "upstream_response"', output)
        self.assertIn('"event": "normalized_response"', output)
        self.assertIn('"status": 200', output)
        self.assertIn("correlation-123", output)
        self.assertIn("[REDACTED]", output)
        self.assertNotIn("test-only-secret-marker", output)
        self.assertNotIn("upstream-token-marker", output)
        self.assertNotIn("cookie-marker", output)

    @patch("lib.metadata_proxy.get_contact_response")
    def test_logs_upstream_and_proxy_error_status(self, get_contact):
        get_contact.return_value = self.response(
            {"detail": "not found"},
            status=404,
        )

        with self.assertLogs("db_logger", level="INFO") as captured:
            with self.assertRaises(MetadataNotFound):
                fetch_contact_metadata("missing", request_id="correlation-404")

        output = "\n".join(captured.output)
        self.assertIn('"status": 404', output)
        self.assertIn('"proxy_status": 404', output)


if __name__ == "__main__":
    unittest.main()
