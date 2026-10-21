"""A local HTTP server that answers like the FPL API from a FakeApi (synthetic season).

Used to test the real network path end to end: the command line in a separate process,
requests, JSON decoding and output encoding, retries and the cache. Only 127.0.0.1.

    with FakeServer(api) as url:           # url = "http://127.0.0.1:<port>/api"
        ...                                # FPLOPT_API_URL=url python -m fplopt ...
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PREFIX = "/api"


class FakeServer:
    def __init__(self, api, failures=None):
        """api: FakeApi. failures: {path: [status, ...]} answered (in order) before the real reply,
        e.g. {'/bootstrap-static/': [503, 429]}. An unknown path gets 404."""
        self.api = api
        self.failures = {k: list(v) for k, v in (failures or {}).items()}
        self.requests = []                       # every path requested, failures included
        self._server = None

    def _handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path[len(PREFIX):] if self.path.startswith(PREFIX) else self.path
                outer.requests.append(path)
                queue = outer.failures.get(path)
                if queue:
                    status = queue.pop(0)
                    self.send_response(status)
                    if status == 429:
                        self.send_header("Retry-After", "0")
                    self.end_headers()
                    return
                try:
                    body = json.dumps(outer.api.respond(path), ensure_ascii=False).encode('utf-8')
                except KeyError:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):        # keep the test output clean
                pass

        return Handler

    def __enter__(self):
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())   # port 0 = any free port
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self._server.server_address[1]}{PREFIX}"

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
