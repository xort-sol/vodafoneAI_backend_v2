import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from shared import results


@pytest.fixture
def sheet(tmp_path, monkeypatch):
    """A fake Google Apps Script endpoint that records what it receives."""
    received, status = [], {"code": 200}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(status["code"])
            self.end_headers()
            self.wfile.write(b"ok" if status["code"] == 200 else b"error")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("RESULTS_WEBHOOK_URL", f"http://127.0.0.1:{server.server_port}/exec")
    monkeypatch.setenv("RESULTS_WEBHOOK_TOKEN", "secret-token")
    yield received, status
    server.shutdown()


def test_result_is_sent_to_sheet_with_token(sheet):
    received, _ = sheet
    results.record_result({"name": "A", "phone": "+1", "outcome": "interested", "room": "web-123"})
    assert received[0]["token"] == "secret-token"
    assert received[0]["outcome"] == "interested"
    assert received[0]["room"] == "web-123"
    assert list(received[0]) == ["token", *results.FIELDS]  # same column order as the sheet header


def test_sheet_failure_keeps_database_copy_and_does_not_raise(sheet, call_rows, caplog):
    _, status = sheet
    status["code"] = 500
    results.record_result({"job_id": "j1", "name": "A", "phone": "+1", "outcome": "interested"})  # must not crash the call
    assert [r["outcome"] for r in call_rows()] == ["interested"]
    assert "could not send result to Google Sheet" in caplog.text
    assert "interested" in caplog.text  # the row is in the logs, so it can be recovered


def test_database_down_still_sends_to_sheet(sheet, monkeypatch):
    received, _ = sheet
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@127.0.0.1:1/none")
    results.record_result({"job_id": "j1", "phone": "+1", "outcome": "interested", "callback_time": "Fri 10am"})
    assert received[0]["callback_time"] == "Fri 10am"
