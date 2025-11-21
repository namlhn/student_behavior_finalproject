from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
from core.database import get_db
from core.fastapi_util import api_response_data
from core.constants import Result
from app import schemas
from db import models
from core.manager import session_manager
import shutil
import os
from core.config import settings

router = APIRouter()

# 1. List Sessions
# Xóa response_model=... để tránh lỗi validation khi trả về object api_response_data


@router.get("/")
def list_sessions(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    sessions = session_manager.get_sessions(db, skip=skip, limit=limit)
    return api_response_data(Result.SUCCESS, sessions)

# 2. Create Session


@router.post("/")
def create_session(session_in: schemas.SessionCreate, db: Session = Depends(get_db)):
    new_session = session_manager.create_session(db, session_in)
    return api_response_data(Result.SUCCESS, new_session)

# 3. Upload Video


@router.post("/{session_id}/upload")
async def upload_video(
    session_id: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    session = session_manager.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Tạo thư mục lưu trữ nếu chưa có
    video_dir = os.path.join(settings.UPLOAD_DIR, "videos")
    os.makedirs(video_dir, exist_ok=True)

    # Lưu file video
    safe_filename = f"session_{session_id}_{file.filename}"
    file_path = os.path.join(video_dir, safe_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Cập nhật đường dẫn DB
    rel_path = f"uploads/videos/{safe_filename}"
    session.video_path = rel_path
    session.status = "queued"
    db.commit()

    # Chạy background task xử lý AI
    # Lưu ý: background task cần import đúng process_video_ai từ session_manager

    background_tasks.add_task(
        session_manager.process_video_ai,
        session_id=session_id,
        video_path=file_path
    )

    return api_response_data(Result.SUCCESS, reply={
        "message": "Upload successful. AI processing started in background.",
        "video_path": rel_path,
        "status": "queued"
    })

# 4. Get Timeline (For Replay)


@router.get("/{session_id}/timeline")
def get_session_timeline(session_id: int, db: Session = Depends(get_db)):
    from sqlalchemy.orm import joinedload
    # Lấy logs hành vi kèm theo logs sinh viên (nested)
    logs = db.query(models.SessionBehaviorLog)\
             .options(joinedload(models.SessionBehaviorLog.students))\
             .filter(models.SessionBehaviorLog.session_id == session_id)\
             .order_by(models.SessionBehaviorLog.timestamp)\
             .all()
    return api_response_data(Result.SUCCESS, logs)

# 5. Get Statistics


@router.get("/{session_id}/stats")
def get_session_stats(session_id: int, db: Session = Depends(get_db)):
    # Thống kê hành vi
    behaviors = db.query(models.SessionBehaviorLog.behavior_type)\
                  .filter(models.SessionBehaviorLog.session_id == session_id).all()

    behavior_counts = {}
    for b in behaviors:
        bt = b[0]
        behavior_counts[bt] = behavior_counts.get(bt, 0) + 1

    # Thống kê cảm xúc (Join bảng)
    emotions = db.query(models.SessionStudentLog.emotion)\
                 .join(models.SessionBehaviorLog)\
                 .filter(models.SessionBehaviorLog.session_id == session_id).all()

    emotion_counts = {}
    for e in emotions:
        et = e[0]
        emotion_counts[et] = emotion_counts.get(et, 0) + 1

    return api_response_data(Result.SUCCESS, reply={
        "behavior_stats": behavior_counts,
        "emotion_stats": emotion_counts
    })

# 6. Update Session


@router.put("/{session_id}")
def update_session(session_id: int, session_in: schemas.SessionUpdate, db: Session = Depends(get_db)):
    session = session_manager.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    updated = session_manager.update_session(db, session, session_in)
    return api_response_data(Result.SUCCESS, updated)

# 7. Delete Session


@router.delete("/{session_id}")
def delete_session(session_id: int, db: Session = Depends(get_db)):
    session = session_manager.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    # Cascade delete related logs (behavior + student logs)
    # 1. Collect behavior log IDs for this session
    behavior_logs = db.query(models.SessionBehaviorLog.id)\
        .filter(models.SessionBehaviorLog.session_id == session_id).all()
    log_ids = [bid for (bid,) in behavior_logs]
    if log_ids:
        # 2. Delete student logs referencing these behavior logs
        db.query(models.SessionStudentLog)\
            .filter(models.SessionStudentLog.behavior_log_id.in_(log_ids))\
            .delete(synchronize_session=False)
        # 3. Delete behavior logs themselves
        db.query(models.SessionBehaviorLog)\
            .filter(models.SessionBehaviorLog.id.in_(log_ids))\
            .delete(synchronize_session=False)
    # 4. Delete the session
    db.delete(session)
    db.commit()
    return api_response_data(Result.SUCCESS, reply={"id": session_id, "deleted_behavior_logs": len(log_ids)})
