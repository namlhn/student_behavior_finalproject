from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from pathlib import Path

test_view_router = APIRouter()
BASE_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@test_view_router.get("/test/behavior")
def page_test_behavior(request: Request):
    return templates.TemplateResponse("pages/test_behavior.html", {
        "request": request,
        "active_page": "test_behavior"
    })


@test_view_router.get("/test/emotion")
def page_test_emotion(request: Request):
    return templates.TemplateResponse("pages/test_emotion.html", {
        "request": request,
        "active_page": "test_emotion"
    })
