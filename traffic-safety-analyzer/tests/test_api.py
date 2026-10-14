"""
Integration tests for the FastAPI endpoints.
"""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from backend.main import app
from backend.database.db import init_db


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
async def setup_db():
    await init_db()


@pytest.mark.anyio
async def test_health_check():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.anyio
async def test_root():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/")
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_dashboard_stats():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/dashboard/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_incidents" in data


@pytest.mark.anyio
async def test_dashboard_analytics():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/dashboard/analytics")
    assert resp.status_code == 200
    data = resp.json()
    assert "violation_distribution" in data
    assert "review_distribution" in data
    assert "timeline" in data


@pytest.mark.anyio
async def test_camera_status():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/camera/status")
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_incidents_list():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/incidents/")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert "total" in data


@pytest.mark.anyio
async def test_incidents_export_csv():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/incidents/export?format=csv")
    assert resp.status_code == 200
    assert "text/csv" in resp.headers.get("content-type", "")
    assert "id,violation_type" in resp.text


@pytest.mark.anyio
async def test_incidents_export_json():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/incidents/export?format=json")
    assert resp.status_code == 200
    assert "application/json" in resp.headers.get("content-type", "")
    assert isinstance(resp.json(), list)


@pytest.mark.anyio
async def test_settings_get_and_patch():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # GET
        resp = await client.get("/api/v1/settings/")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_name" in data

        # PATCH
        patch_resp = await client.patch("/api/v1/settings/", json={"confidence_threshold": 0.45})
        assert patch_resp.status_code == 200
        patch_data = patch_resp.json()
        assert patch_data["settings"]["confidence_threshold"] == 0.45


@pytest.mark.anyio
async def test_monitoring_status():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/monitoring/status")
    assert resp.status_code == 200
    assert "camera_running" in resp.json()
