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


@pytest.mark.anyio
async def test_incidents_batch_review_and_bulk_delete():
    from backend.database.db import AsyncSessionLocal
    from backend.database.models import Incident
    from datetime import datetime, timezone
    import uuid

    id1, id2 = str(uuid.uuid4()), str(uuid.uuid4())
    async with AsyncSessionLocal() as session:
        inc1 = Incident(
            id=id1,
            violation_type="no_helmet",
            event_timestamp=datetime.now(timezone.utc),
            camera_source="test_cam",
            confidence=0.91,
            detection_metadata={},
            review_status="pending",
        )
        inc2 = Incident(
            id=id2,
            violation_type="triple_riding",
            event_timestamp=datetime.now(timezone.utc),
            camera_source="test_cam",
            confidence=0.87,
            detection_metadata={},
            review_status="pending",
        )
        session.add_all([inc1, inc2])
        await session.commit()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Batch review
        review_resp = await client.post(
            "/api/v1/incidents/batch-review",
            json={
                "incident_ids": [id1, id2],
                "status": "confirmed",
                "notes": "Verified in batch",
            },
        )
        assert review_resp.status_code == 200
        assert review_resp.json()["updated_count"] == 2
        assert review_resp.json()["status"] == "confirmed"

        # Verify status updated
        get_resp = await client.get(f"/api/v1/incidents/{id1}")
        assert get_resp.status_code == 200
        assert get_resp.json()["review_status"] == "confirmed"
        assert get_resp.json()["reviewer_notes"] == "Verified in batch"

        # Bulk delete
        del_resp = await client.post(
            "/api/v1/incidents/bulk-delete",
            json={"incident_ids": [id1, id2]},
        )
        assert del_resp.status_code == 200
        assert del_resp.json()["deleted_count"] == 2

        # Verify deleted
        get_deleted = await client.get(f"/api/v1/incidents/{id1}")
        assert get_deleted.status_code == 404


@pytest.mark.anyio
async def test_monitoring_recordings_and_snapshots():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        rec_resp = await client.get("/api/v1/monitoring/recordings")
        assert rec_resp.status_code == 200
        assert "recordings" in rec_resp.json()

        snap_resp = await client.get("/api/v1/monitoring/snapshots")
        assert snap_resp.status_code == 200
        assert "snapshots" in snap_resp.json()


@pytest.mark.anyio
async def test_camera_upload_invalid_and_valid():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Invalid file type
        bad_resp = await client.post(
            "/api/v1/camera/upload",
            files={"file": ("test.txt", b"plain text", "text/plain")},
        )
        assert bad_resp.status_code == 400

        # Valid video upload
        good_resp = await client.post(
            "/api/v1/camera/upload",
            files={"file": ("sample.mp4", b"\x00\x00\x00\x18ftypmp42", "video/mp4")},
        )
        assert good_resp.status_code == 200
        data = good_resp.json()
        assert data["filename"] == "sample.mp4"
        assert "uploads/sample.mp4" in data["source"]

