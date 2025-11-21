from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from pathlib import Path

dashboard_router = APIRouter()

# Ensure path points to the correct templates folder
BASE_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@dashboard_router.get("/dashboard")
async def view_dashboard(request: Request):
    """
    Renders the main admin dashboard.
    """
    return templates.TemplateResponse("pages/dashboard.html", {
        "request": request,
        "active_page": "dashboard"  # Used for highlighting sidebar menu
    })