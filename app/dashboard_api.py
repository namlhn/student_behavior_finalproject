from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from core.database import get_db
from core.fastapi_util import api_response_data
from core.constants import Result
from db import models
import datetime

router = APIRouter()


@router.get("/stats/general")
def get_general_stats(db: Session = Depends(get_db)):
    """
    API trả về số liệu toàn hệ thống cho Dashboard
    """
    # 1. Các con số cơ bản (Cards)
    total_students = db.query(models.Student).count()
    total_sessions = db.query(models.ClassSession).count()

    # Tổng số hành vi đã detect được
    total_behaviors = db.query(models.SessionBehaviorLog).count()

    # 2. Thống kê Hành vi (Pie Chart)
    behavior_stats = db.query(
        models.SessionBehaviorLog.behavior_type,
        func.count(models.SessionBehaviorLog.id)
    ).group_by(models.SessionBehaviorLog.behavior_type).all()

    # 3. Thống kê Cảm xúc (Bar Chart)
    emotion_stats = db.query(
        models.SessionStudentLog.emotion,
        func.count(models.SessionStudentLog.id)
    ).group_by(models.SessionStudentLog.emotion).all()

    # 4. Danh sách buổi học gần đây (Recent Activity)
    recent_sessions = db.query(models.ClassSession)\
        .order_by(models.ClassSession.created_at.desc())\
        .limit(5).all()

    # Format dữ liệu recent sessions
    recent_list = []
    for s in recent_sessions:
        recent_list.append({
            "id": s.id,
            "class_name": s.class_name,
            "subject_name": s.subject_name,
            "date": s.session_date.strftime("%Y-%m-%d") if s.session_date else "N/A",
            "status": s.status,
            "detections": s.total_detections or 0
        })

    return api_response_data(Result.SUCCESS, reply={
        "counts": {
            "students": total_students,
            "sessions": total_sessions,
            "behaviors": total_behaviors
        },
        "charts": {
            "behavior": dict(behavior_stats),
            "emotion": dict(emotion_stats)
        },
        "recent_sessions": recent_list
    })
