from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from core.database import get_db
from core.fastapi_util import api_response_data
from core.constants import Result
from db import models
from typing import List, Dict, Any

router = APIRouter()

# --- 1. API Lấy danh sách Lớp & Môn (Để fill vào Dropdown lọc) ---


@router.get("/options/classes")
def get_class_options(db: Session = Depends(get_db)):
    """
    Trả về danh sách duy nhất các cặp (Lớp, Môn) đang có trong DB.
    Ví dụ: [{"class": "12A1", "subject": "Toán"}, ...]
    """
    # Query lấy các cặp class_name, subject_name không trùng lặp
    results = db.query(
        models.ClassSession.class_name,
        models.ClassSession.subject_name
    ).distinct().all()

    options = [{"class": r[0], "subject": r[1]}
               for r in results if r[0] and r[1]]
    return api_response_data(Result.SUCCESS, options)


# --- 2. API Báo cáo Tổng hợp (Class-Level Report) ---
@router.get("/class/summary")
def get_class_summary_report(
    class_name: str,
    subject_name: str,
    db: Session = Depends(get_db)
):
    """
    Tổng hợp dữ liệu từ TẤT CẢ các buổi học (Sessions) của một Lớp + Môn cụ thể.
    Trả về:
    - Meta: Thông tin chung (Tổng số buổi, ...)
    - Trend: Diễn biến điểm số/hành vi qua từng ngày.
    - Students: Bảng xếp hạng và chi tiết từng sinh viên.
    """

    # A. Lấy danh sách các buổi học đã hoàn thành (Status = completed)
    sessions = db.query(models.ClassSession).filter(
        models.ClassSession.class_name == class_name,
        models.ClassSession.subject_name == subject_name,
        models.ClassSession.status == 'completed'
    ).order_by(models.ClassSession.session_date).all()

    if not sessions:
        return api_response_data(Result.SUCCESS, reply=None)

    session_ids = [s.id for s in sessions]
    total_sessions_count = len(sessions)

    # B. Tính toán Trend (Diễn biến qua từng buổi)
    # Mục tiêu: Vẽ biểu đồ đường thể hiện "Độ sôi nổi" của lớp qua các ngày
    session_trend = []

    for s in sessions:
        # Đếm số lượng từng loại hành vi trong buổi này
        # Kết quả ví dụ: [('hand-raising', 5), ('writing', 20), ...]
        b_counts = db.query(
            models.SessionBehaviorLog.behavior_type,
            func.count(models.SessionBehaviorLog.id)
        ).filter(
            models.SessionBehaviorLog.session_id == s.id
        ).group_by(models.SessionBehaviorLog.behavior_type).all()

        b_dict = dict(b_counts)

        # Công thức tính điểm Engagement (có thể tùy chỉnh trọng số)
        # Ví dụ: Giơ tay (x2 điểm), Viết/Đọc (x1 điểm)
        score = (b_dict.get('hand-raising', 0) * 2) + \
                (b_dict.get('writing', 0) * 1) + \
                (b_dict.get('reading', 0) * 1) + \
                (b_dict.get('discuss', 0) * 1.5)

        session_trend.append({
            "session_id": s.id,
            "date": s.session_date.strftime("%d/%m/%Y") if s.session_date else "N/A",
            "score": score,
            "total_detections": s.total_detections or 0
        })

    # C. Thống kê chi tiết từng Sinh viên (Aggregate Student Stats)
    # Cần biết: Sinh viên đi học bao nhiêu buổi? Cảm xúc chủ đạo là gì? Hành vi thường làm là gì?

    # Lấy toàn bộ log sinh viên trong danh sách session_ids này
    # Join bảng StudentLog -> BehaviorLog để lấy thông tin session_id và behavior_type
    raw_logs = db.query(
        models.SessionStudentLog.student_name,
        models.SessionStudentLog.emotion,
        models.SessionBehaviorLog.behavior_type,
        models.SessionBehaviorLog.session_id
    ).join(models.SessionBehaviorLog)\
     .filter(models.SessionBehaviorLog.session_id.in_(session_ids))\
     .filter(models.SessionStudentLog.student_name != "Unknown")\
     .all()

    # Xử lý dữ liệu thô bằng Python (Aggregation)
    # Cấu trúc stu_stats:
    # {
    #   "Nguyen Van A": {
    #       "sessions": {1, 5, 8}, // Set các session_id đã xuất hiện (để tính điểm danh)
    #       "emotions": {"happy": 10, "neutral": 50},
    #       "behaviors": {"reading": 30, "writing": 20}
    #   }
    # }
    stu_stats: Dict[str, Dict[str, Any]] = {}

    for row in raw_logs:
        name, emo, beh, sid = row

        if name not in stu_stats:
            stu_stats[name] = {
                "sessions_attended": set(),
                "emotions": {},
                "behaviors": {}
            }

        # Ghi nhận sự xuất hiện
        stu_stats[name]["sessions_attended"].add(sid)

        # Cộng dồn cảm xúc
        stu_stats[name]["emotions"][emo] = stu_stats[name]["emotions"].get(
            emo, 0) + 1

        # Cộng dồn hành vi
        stu_stats[name]["behaviors"][beh] = stu_stats[name]["behaviors"].get(
            beh, 0) + 1

    # Format danh sách kết quả trả về
    student_list = []
    for name, data in stu_stats.items():
        # Tìm cảm xúc phổ biến nhất (Dominant Emotion)
        dom_emo = "-"
        if data["emotions"]:
            dom_emo = max(data["emotions"], key=data["emotions"].get)

        # Tìm hành vi phổ biến nhất (Frequent Behavior)
        dom_beh = "-"
        if data["behaviors"]:
            dom_beh = max(data["behaviors"], key=data["behaviors"].get)

        # Tính số buổi tham gia (Dựa trên việc AI có nhận diện được mặt trong buổi đó không)
        attended_count = len(data["sessions_attended"])
        attendance_rate = 0
        if total_sessions_count > 0:
            attendance_rate = round(
                (attended_count / total_sessions_count) * 100, 1)

        # Tổng số lần được AI detect (Activity Level)
        total_logs = sum(data["emotions"].values())

        student_list.append({
            "name": name,
            "attendance_rate": attendance_rate,     # Tỷ lệ % tham gia
            "attended_count": attended_count,       # Số buổi có mặt
            "dominant_emotion": dom_emo,            # Cảm xúc thường thấy
            "dominant_behavior": dom_beh,           # Hành vi thường làm
            # Điểm hoạt động (số lần detect)
            "total_logs": total_logs
        })

    # Sắp xếp danh sách theo tên A-Z
    student_list.sort(key=lambda x: x['name'])

    return api_response_data(Result.SUCCESS, reply={
        "meta": {
            "class_name": class_name,
            "subject_name": subject_name,
            "total_sessions": total_sessions_count
        },
        "trend": session_trend,
        "students": student_list
    })
