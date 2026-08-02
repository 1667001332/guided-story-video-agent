from __future__ import annotations

from guided_story_agent.v2 import HttpResponse, MockHttpTransport


def test_mock_transport_records_requests_without_network() -> None:
    transport = MockHttpTransport([HttpResponse(200, json_data={"ok": True})])
    response = transport.request("GET", "mock://health", headers={"Authorization": "TEST_PROVIDER_SECRET_123"})
    assert response.status_code == 200
    assert transport.request_count == 1
    assert transport.real_network_calls == 0
    assert transport.requests[0].headers["Authorization"] == "TEST_PROVIDER_SECRET_123"
