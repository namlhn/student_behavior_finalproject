from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from pathlib import Path
import random

from core.database import get_db
from core.manager import student_manager

# Setup Router & Template
student_view_router = APIRouter()
BASE_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@student_view_router.get("/students/add")
async def add_student_page(request: Request):
    return templates.TemplateResponse("pages/add_student.html", {
        "request": request,
        "active_page": "students"
    })


@student_view_router.get("/students")
async def list_students_page(request: Request):
    return templates.TemplateResponse("pages/students.html", {
        "request": request,
        "active_page": "students"
    })


@student_view_router.get("/students/{student_id}/embedding")
async def embedding_page(student_id: int, request: Request, db: Session = Depends(get_db)):
    # Lấy thông tin sinh viên qua Manager
    student = student_manager.get_student(db, student_id)
    if not student:
        return templates.TemplateResponse("pages/404.html", {"request": request})

    return templates.TemplateResponse("pages/embedding.html", {
        "request": request,
        "student": student,
        "active_page": "students"
    })


@student_view_router.get("/face-detect")
async def face_detect_page(request: Request):
    return templates.TemplateResponse("pages/face_detect.html", {
        "request": request,
        "active_page": "face_detect"
    })
