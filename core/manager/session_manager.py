import cv2
import numpy as np
from sqlalchemy.orm import Session
from db import models
from app import schemas
from core.ai_loader import ai_engine
from db.vector_db import vector_db_instance
import json
from core.database import SessionLocal
# --- CRUD OPERATIONS ---


def create_session(db: Session, session_in: schemas.SessionCreate):
    """Tạo mới một buổi học trong CSDL"""
    # Chuyển đổi schema Pydantic thành model SQLAlchemy
    db_session = models.ClassSession(**session_in.dict())
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session


def get_session(db: Session, session_id: int):
    """Lấy thông tin chi tiết một buổi học"""
    return db.query(models.ClassSession).filter(models.ClassSession.id == session_id).first()


def get_sessions(db: Session, skip: int = 0, limit: int = 100):
    """Lấy danh sách các buổi học (có phân trang)"""
    return db.query(models.ClassSession).order_by(models.ClassSession.created_at.desc()).offset(skip).limit(limit).all()


def update_session(db: Session, session: models.ClassSession, session_in: schemas.SessionUpdate):
    """Cập nhật thông tin một buổi học. Chỉ ghi đè các trường được gửi lên."""
    data = session_in.model_dump(exclude_unset=True)
    for field, value in data.items():
        setattr(session, field, value)
    db.commit()
    db.refresh(session)
    return session

# --- AI LOGIC & HELPERS ---


def _get_student_from_vector_id(db: Session, vector_id: int):
    """Tìm sinh viên dựa trên ID vector từ FAISS"""
    photo = db.query(models.StudentPhoto).filter(
        models.StudentPhoto.faiss_vector_id == vector_id).first()
    if photo:
        return db.query(models.Student).filter(models.Student.id == photo.student_id).first()
    return None


def _predict_emotion(face_img):
    """Dự đoán cảm xúc từ ảnh crop khuôn mặt (Giả lập hoặc gọi model thật)"""
    # Nếu chưa load model, trả về unknown
    if ai_engine.emotion_model is None:
        return "neutral"

    try:
        # Ở đây bạn cần code tiền xử lý ảnh cho ResNet (Resize, ToTensor...)
        # Để đơn giản, ta demo logic trả về kết quả
        # Thực tế cần: img -> transform -> model -> prediction
        return "neutral"
    except Exception:
        return "unknown"


def process_video_ai(session_id: int, video_path: str):
    """
    Hàm xử lý video chạy ngầm.
    Tự quản lý DB Session để tránh lỗi 'Session closed' khi API return.
    """
    # [MỚI] Tự tạo session riêng
    db = SessionLocal()

    try:
        session = get_session(db, session_id)
        if not session:
            print(f"Session {session_id} not found.")
            return

        # Cập nhật trạng thái: Đang xử lý
        session.status = "processing"
        db.commit()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        process_interval = int(fps) if fps > 0 else 30
        frame_count = 0

        print(f"Started AI processing for session {session_id}...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % process_interval == 0:
                current_time = round(frame_count / (fps if fps > 0 else 30), 2)

                # --- 1. YOLO Detect Behavior ---
                detected_behaviors = []
                if ai_engine.behavior_model:
                    results = ai_engine.behavior_model(frame, verbose=False)
                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu(
                            ).numpy().astype(int)
                            conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            label = ai_engine.behavior_model.names[cls_id]
                            detected_behaviors.append({
                                'label': label,
                                'box': [x1, y1, x2, y2],
                                'conf': conf
                            })

                # --- 2. Xử lý từng hành vi ---
                for b_item in detected_behaviors:
                    bx1, by1, bx2, by2 = b_item['box']
                    behavior_label = b_item['label']

                    # Validate box coordinates
                    h_img, w_img = frame.shape[:2]
                    bx1, by1 = max(0, bx1), max(0, by1)
                    bx2, by2 = min(w_img, bx2), min(h_img, by2)

                    if bx2 <= bx1 or by2 <= by1:
                        continue

                    # Lưu SessionBehaviorLog (Parent)
                    behavior_log = models.SessionBehaviorLog(
                        session_id=session_id,
                        timestamp=current_time,
                        behavior_type=behavior_label,
                        bbox=f"{bx1},{by1},{bx2},{by2}"
                    )
                    db.add(behavior_log)
                    db.flush()  # Flush để lấy ID

                    # Crop & Detect Face
                    behavior_crop = frame[by1:by2, bx1:bx2]

                    faces = []
                    if ai_engine.identity_model and behavior_crop.size > 0:
                        faces = ai_engine.identity_model.get(behavior_crop)

                    for face in faces:
                        fx1, fy1, fx2, fy2 = face.bbox.astype(int)

                        # Convert Local -> Global coords
                        g_fx1, g_fy1 = bx1 + fx1, by1 + fy1
                        g_fx2, g_fy2 = bx1 + fx2, by1 + fy2

                        # Identify
                        student_id = 0
                        student_name = "Unknown"
                        if face.embedding is not None:
                            emb_arr = np.array(
                                [face.embedding], dtype='float32')
                            vec_id, sim = vector_db_instance.search_embedding(
                                emb_arr)
                            if vec_id:
                                stu = _get_student_from_vector_id(db, vec_id)
                                if stu:
                                    student_id = stu.id
                                    student_name = stu.name

                        # Emotion placeholder
                        emotion_label = "neutral"

                        # Lưu SessionStudentLog (Child)
                        student_log = models.SessionStudentLog(
                            behavior_log_id=behavior_log.id,
                            student_id=student_id,
                            student_name=student_name,
                            emotion=emotion_label,
                            face_bbox=f"{g_fx1},{g_fy1},{g_fx2},{g_fy2}"
                        )
                        db.add(student_log)

                db.commit()

            frame_count += 1

        # Hoàn tất
        session.status = "completed"
        count = db.query(models.SessionBehaviorLog).filter(
            models.SessionBehaviorLog.session_id == session_id).count()
        session.total_detections = count
        db.commit()
        print(f"Session {session_id} processing completed.")

    except Exception as e:
        print(f"Error processing session {session_id}: {e}")
        # Cần rollback nếu lỗi để tránh treo transaction
        db.rollback()
        # Mở session mới hoặc dùng session hiện tại để update status failed
        try:
            err_session = get_session(db, session_id)
            if err_session:
                err_session.status = "failed"
                db.commit()
        except:
            pass
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        # [QUAN TRỌNG] Đóng kết nối
        db.close()
