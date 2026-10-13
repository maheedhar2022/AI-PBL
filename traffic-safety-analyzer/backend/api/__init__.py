from backend.api.camera_routes import router as camera_router
from backend.api.monitoring_routes import router as monitoring_router
from backend.api.incident_routes import router as incident_router
from backend.api.dashboard_routes import router as dashboard_router
from backend.api.evaluation_routes import router as evaluation_router
from backend.api.settings_routes import router as settings_router
from backend.api.ws_routes import router as ws_router

__all__ = [
    "camera_router",
    "monitoring_router",
    "incident_router",
    "dashboard_router",
    "evaluation_router",
    "settings_router",
    "ws_router",
]

