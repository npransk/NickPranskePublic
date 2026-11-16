# Showcase copy of services/api/tests/tracing_probe.py from the private mahjong-companion repo. Here to be read, not run.

"""Run one strategist ask with the real Anthropic and Langfuse SDKs against a stand-in server,
and print what Langfuse would have received. Run by test_tracing_e2e.py in its own process,
because a live Langfuse client sets process-wide tracing state.

One local server plays both parts: Claude's Messages API and Langfuse's ingestion endpoints.

    python tracing_probe.py <card.toml> [--unfiltered]

``--unfiltered`` exports what Langfuse would export by default, to show what the app's filter
removes.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPLY = {
    "id": "msg_probe",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5-5",
    "content": [
        {
            "type": "tool_use",
            "id": "toolu_probe",
            "name": "submit_pick",
            "input": {"hand_id": "1999-LIKE-1", "reason": "Go for like 3s: you hold nine."},
        }
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 50,
    },
}
received: dict[str, list] = {"spans": [], "scores": [], "claude_requests": []}


class StandIn(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(404, {"error": "NotFound", "message": "Nothing registered here"})

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        if self.path.startswith("/v1/messages"):
            received["claude_requests"].append(json.loads(body)["model"])
            self._send(200, REPLY)
        elif self.path.endswith("/otel/v1/traces"):
            received["spans"].append(body)
            self._send(200, {})
        elif self.path.endswith("/api/public/scores"):
            received["scores"].extend(json.loads(body))
            self._send(200, {})
        else:
            self._send(200, {})


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), StandIn)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    address = f"http://127.0.0.1:{server.server_port}"
    os.environ.update(
        LANGFUSE_PUBLIC_KEY="pk-lf-probe",
        LANGFUSE_SECRET_KEY="sk-lf-probe",
        LANGFUSE_BASE_URL=address,
        LANGFUSE_TRACING_ENVIRONMENT="selftest",
        LANGFUSE_RELEASE="abc1234",
        NO_PROXY="127.0.0.1",
        no_proxy="127.0.0.1",
    )
    for name in ("LANGFUSE_PUBLIC_KEY_MJ", "LANGFUSE_SECRET_KEY_MJ", "OTEL_SERVICE_NAME"):
        os.environ.pop(name, None)

    import anthropic
    from langfuse.span_filter import is_default_export_span
    from mahjong_api import observability, strategist
    from mahjong_engine.card import load_card
    from mahjong_engine.game import Game
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    if "--unfiltered" in sys.argv:
        observability._should_export = is_default_export_span
    card = load_card(Path(sys.argv[1]))
    game = Game(players=2, dealer=2)
    hand = ["flower", "flower", "3 bam", "3 bam", "3 bam", "3 crak", "3 crak", "3 crak"]
    game.add({"type": "deal", "tiles": hand + ["3 dot", "3 dot", "7 crak", "9 bam", "6 dot"]})
    claude = anthropic.Anthropic(api_key="sk-ant-probe", base_url=address, max_retries=0)
    pick = strategist.ask(claude, card, game.state(), "probe-game")
    observability.langfuse_client().shutdown()

    spans, resource = [], {}
    for body in received["spans"]:
        for resource_spans in ExportTraceServiceRequest.FromString(body).resource_spans:
            for attribute in resource_spans.resource.attributes:
                resource[attribute.key] = attribute.value.string_value
            for scope_spans in resource_spans.scope_spans:
                for span in scope_spans.spans:
                    attributes = {a.key: a.value.string_value for a in span.attributes}
                    spans.append(
                        {
                            "scope": scope_spans.scope.name,
                            "name": span.name,
                            "type": attributes.get("langfuse.observation.type"),
                            "environment": attributes.get("langfuse.environment"),
                            "session": attributes.get("session.id"),
                            "version": attributes.get("langfuse.version"),
                            "tags": [
                                v.string_value
                                for a in span.attributes
                                if a.key == "langfuse.trace.tags"
                                for v in a.value.array_value.values
                            ],
                        }
                    )
    scores = {score["name"]: score["value"] for score in received["scores"]}
    print(
        json.dumps(
            {
                "pick": pick.hand_id,
                "claude_requests": received["claude_requests"],
                "spans": spans,
                "resource": resource,
                "scores": scores,
            }
        )
    )


if __name__ == "__main__":
    main()
