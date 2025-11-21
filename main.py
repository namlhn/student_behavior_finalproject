from fastapi import FastAPI
import os
from pathlib import Path
from core.database import engine, Base
from core.config import settings
from fastapi.staticfiles import StaticFiles
from core.middleware import apply_middlewares
from app.dashboard_view import dashboard_router
from fastapi.responses import RedirectResponse
from app.student_view import student_view_router
from app.session_api import router as session_router
from app.session_view import session_view_router
from app.student_api import router as student_api_router
from app.test_api import router as test_api_router
from app.test_view import test_view_router
from app.report_api import router as report_api_router
from app.report_view import report_view_router
from app.dashboard_api import router as dashboard_api_router
try:
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully (if not exist).")
except Exception as e:
    print(f"Error creating database tables: {e}")
    print("Please ensure MySQL server is running and database is created.")

app = FastAPI(title="Student Behavior AI Web",
              docs_url="/docs", redoc_url="/redoc")

# Middlewares (CORS, Request ID)
apply_middlewares(app, settings)


# Uploads only (no legacy UI)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# Images (logo, banner)
images_dir = Path(__file__).resolve().parent / "assets" / "images"
os.makedirs(images_dir, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")

# Static files (CSS, JS)
app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "assets")), name="static")
# Routers

app.include_router(dashboard_router)
app.include_router(student_view_router)
app.include_router(session_view_router)
app.include_router(test_view_router)
app.include_router(session_router, prefix="/api/sessions",
                   tags=["Class Sessions"])
app.include_router(student_api_router,
                   prefix="/api/students", tags=["Students"])
app.include_router(test_api_router, prefix="/api/test", tags=["Test Models"])
app.include_router(report_api_router, prefix="/api/reports", tags=["Reports"])
app.include_router(report_view_router)
app.include_router(dashboard_api_router,
                   prefix="/api/dashboard", tags=["Dashboard"])

# add / redirect to /dashboard


@app.get("/")
async def root():
    return RedirectResponse(url="/dashboard")
