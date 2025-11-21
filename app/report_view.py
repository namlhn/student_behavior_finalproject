from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
from pathlib import Path
from sqlalchemy.orm import Session
from core.database import get_db
from core.manager import session_manager

report_view_router = APIRouter()
BASE_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@report_view_router.get("/reports/class-summary")
def view_class_summary_report(request: Request):
    return templates.TemplateResponse("pages/class_report.html", {
        "request": request,
        "active_page": "class_report"
    })
