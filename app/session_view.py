from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
from pathlib import Path
from sqlalchemy.orm import Session
from core.database import get_db
from core.manager import session_manager
from app import schemas

session_view_router = APIRouter()
BASE_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@session_view_router.get("/sessions")
def view_sessions_list(request: Request):
    """Trang danh sách các buổi học"""
    return templates.TemplateResponse("pages/sessions.html", {
        "request": request,
        "active_page": "sessions"
    })


@session_view_router.get("/sessions/{session_id}")
def view_session_detail(request: Request, session_id: int, db: Session = Depends(get_db)):
    """Trang chi tiết: Upload video & Xem kết quả"""
    session = session_manager.get_session(db, session_id)
    if not session:
        return templates.TemplateResponse("pages/sessions.html", {
            "request": request,
            "active_page": "sessions"
        })
    # Dùng Pydantic để chuẩn hóa và chuyển sang JSON string (ISO format) cho JS
    session_out = schemas.SessionOut.model_validate(
        session, from_attributes=True)
    session_json = session_out.model_dump_json()
    return templates.TemplateResponse("pages/session_detail.html", {
        "request": request,
        "active_page": "sessions",
        "session": session_out.model_dump(),   # server-side render (dict)
        "session_json": session_json            # client-side JSON parse
    })
