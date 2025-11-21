import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_
from datetime import datetime, date
from db import models
from app import schemas
from fastapi import UploadFile
from sqlalchemy.orm import Session
from core.config import settings
from db.vector_db import vector_db_instance
from core.ai_loader import ai_engine
import os
import cv2


def get_students(db: Session, search: str = None, limit: int = 50):
    query = db.query(models.Student)
    if search:
        query = query.filter(
            or_(
                models.Student.name.ilike(f"%{search}%"),
                models.Student.student_code.ilike(f"%{search}%"),
                models.Student.class_name.ilike(f"%{search}%"),
            )
        )
    return query.limit(limit).all()


def get_student(db: Session, student_id: int):
    return db.query(models.Student).filter(models.Student.id == student_id).first()


def get_student_by_email(db: Session, email: str):
    return db.query(models.Student).filter(models.Student.email == email).first()


def create_student(db: Session, student: schemas.StudentCreate):
    # Nếu không có student_code, tự động sinh (VD: SV + timestamp)
    if not student.student_code:
        import time
        student.student_code = f"SV{int(time.time())}"

    db_student = models.Student(**student.dict())
    db.add(db_student)
    db.commit()
    db.refresh(db_student)
    return db_student


def update_student(db: Session, student_id: int, student_data: schemas.StudentUpdate):
    db_student = get_student(db, student_id)
    if not db_student:
        return None

    # Chỉ cập nhật các trường được gửi lên (exclude_unset=True)
    update_data = student_data.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_student, key, value)

    db.add(db_student)
    db.commit()
    db.refresh(db_student)
    return db_student


async def add_student_faces(db: Session, student_id: int, files: list[UploadFile]):
    """
    Xử lý upload ảnh: Đọc byte trực tiếp -> InsightFace -> Vector DB -> MySQL
    """
    student = get_student(db, student_id)
    if not student:
        raise Exception("Student not found")

    # Tạo thư mục để lưu file ảnh gốc (chỉ để hiển thị web)
    save_dir = os.path.join(settings.UPLOAD_DIR, "faces", str(student_id))
    os.makedirs(save_dir, exist_ok=True)

    count_success = 0
    errors = []

    # Kiểm tra AI Engine đã load chưa
    if ai_engine.identity_model is None:
        return {
            "success_count": 0,
            "total_embeddings": student.face_embedding_count or 0,
            "errors": ["Hệ thống AI chưa sẵn sàng (Model not loaded)"]
        }

    def _safe_bbox(bbox, w, h):
        """Clamp and validate bbox array-like to (x1,y1,x2,y2). Returns None if invalid."""
        try:
            if bbox is None or len(bbox) < 4:
                return None
            x1, y1, x2, y2 = bbox[:4]
            # Ensure numeric
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            # Fix inverted coordinates
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            # Clamp
            x1 = max(0, min(int(x1), w-1))
            y1 = max(0, min(int(y1), h-1))
            x2 = max(0, min(int(x2), w-1))
            y2 = max(0, min(int(y2), h-1))
            # Guarantee at least 2px size
            if x2 <= x1:
                x2 = min(w-1, x1+1)
            if y2 <= y1:
                y2 = min(h-1, y1+1)
            return x1, y1, x2, y2
        except Exception:
            return None

    for file in files:
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

            if img is None:
                errors.append(
                    f"{file.filename}: Lỗi định dạng ảnh hoặc file hỏng")
                continue

            # Chuẩn hoá về BGR 3 kênh nếu là ảnh có alpha hoặc grayscale
            if len(img.shape) == 2:  # grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:  # BGRA -> BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            h, w = img.shape[:2]
            if h < 64 or w < 64:
                errors.append(
                    f"{file.filename}: Ảnh quá nhỏ ({w}x{h}), cần tối thiểu 64x64")
                continue

            # Lưu file gốc
            file_path = os.path.join(save_dir, file.filename)
            with open(file_path, "wb") as f:
                f.write(contents)

            # Nhận diện khuôn mặt
            faces = []
            try:
                faces = ai_engine.identity_model.get(img)
            except Exception as fe:
                errors.append(f"{file.filename}: Lỗi gọi model ({str(fe)})")
                continue
            if not faces:
                errors.append(f"{file.filename}: Không tìm thấy khuôn mặt nào")
                continue

            # Lọc các face có bbox hợp lệ (>=4 phần tử)
            valid_faces = []
            for f in faces:
                bbox = getattr(f, 'bbox', None)
                if bbox is None or len(bbox) < 4:
                    continue
                # make sure bbox indexes are safe
                h_img, w_img = img.shape[:2]
                clamped = _safe_bbox(bbox, w_img, h_img)
                if clamped:
                    # Attach clamped bbox for area calculation
                    f._safe_clamped_bbox = clamped  # dynamic attribute
                    valid_faces.append(f)
            if not valid_faces:
                errors.append(
                    f"{file.filename}: Dữ liệu khuôn mặt không hợp lệ (bbox lỗi)")
                continue

            # Chọn khuôn mặt lớn nhất theo diện tích bbox an toàn
            try:
                face = max(valid_faces, key=lambda x: (
                    x._safe_clamped_bbox[2] - x._safe_clamped_bbox[0]) * (x._safe_clamped_bbox[3] - x._safe_clamped_bbox[1]))
            except Exception as fe:
                errors.append(
                    f"{file.filename}: Lỗi tính diện tích bbox ({str(fe)})")
                continue

            embedding = getattr(face, 'embedding', None)
            if embedding is None:
                errors.append(
                    f"{file.filename}: Chất lượng khuôn mặt kém, không lấy được đặc trưng")
                continue

            # Ensure embedding is correct shape (1, dim) for FAISS
            embedding_arr = np.array(embedding, dtype='float32')
            if embedding_arr.ndim == 1:
                embedding_arr = embedding_arr.reshape(1, -1)
            elif embedding_arr.ndim == 2 and embedding_arr.shape[0] != 1:
                # Take first vector if multiple returned
                embedding_arr = embedding_arr[0:1]
            if embedding_arr.shape[1] != settings.EMBEDDING_DIM:
                errors.append(
                    f"{file.filename}: Sai kích thước embedding ({embedding_arr.shape})")
                continue

            # Lưu embedding vào vector DB và nhận faiss_id
            try:
                faiss_id = vector_db_instance.add_embedding(
                    student_id, embedding_arr)
            except Exception as ve:
                errors.append(
                    f"{file.filename}: Lỗi lưu embedding ({str(ve)})")
                continue

            rel_path = f"uploads/faces/{student_id}/{file.filename}"
            new_photo = models.StudentPhoto(
                student_id=student_id, photo_path=rel_path, faiss_vector_id=faiss_id)
            db.add(new_photo)

            if not student.photo_path:
                student.photo_path = rel_path
                db.add(student)

            count_success += 1

        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            errors.append(f"{file.filename}: Lỗi xử lý ({str(e)})")

    # 7. Commit thay đổi vào DB
    if count_success > 0:
        student.face_embedding_count = (
            student.face_embedding_count or 0) + count_success
        db.commit()
        db.refresh(student)

    return {
        "success_count": count_success,
        "total_embeddings": student.face_embedding_count,
        "errors": errors
    }


def get_student_photos(db: Session, student_id: int):
    return db.query(models.StudentPhoto).filter(models.StudentPhoto.student_id == student_id).order_by(models.StudentPhoto.created_at.desc()).all()


def delete_all_embeddings(db: Session, student_id: int):
    student = get_student(db, student_id)
    if not student:
        return {"deleted": 0, "remaining": 0, "message": "Student not found"}
    removed = vector_db_instance.delete_embeddings_for_student(student_id)
    student.face_embedding_count = 0
    db.add(student)
    db.commit()
    db.refresh(student)
    return {"deleted": int(removed), "remaining": student.face_embedding_count}


def _extract_embedding_from_image(img):
    if ai_engine.identity_model is None:
        return None
    faces = ai_engine.identity_model.get(img)
    if not faces:
        return None
    # pick largest valid face
    valid = [f for f in faces if getattr(f, 'bbox', None) is not None and len(
        f.bbox) >= 4 and getattr(f, 'embedding', None) is not None]
    if not valid:
        return None
    face = max(valid, key=lambda x: (
        x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    emb = getattr(face, 'embedding', None)
    if emb is None:
        return None
    emb_arr = np.array(emb, dtype='float32')
    if emb_arr.ndim == 1:
        emb_arr = emb_arr.reshape(1, -1)
    elif emb_arr.ndim == 2 and emb_arr.shape[0] != 1:
        emb_arr = emb_arr[0:1]
    return emb_arr


def rebuild_student_embeddings(db: Session, student_id: int):
    student = get_student(db, student_id)
    if not student:
        return {"readded": 0, "total": 0, "message": "Student not found"}

    # Clear existing embeddings
    vector_db_instance.delete_embeddings_for_student(student_id)
    student.face_embedding_count = 0
    db.add(student)

    photos = get_student_photos(db, student_id)
    base_dir = settings.UPLOAD_DIR
    readded = 0
    for p in photos:
        abs_path = os.path.join(base_dir, os.path.basename(
            base_dir), *p.photo_path.split('/')[1:]) if not os.path.isabs(p.photo_path) else p.photo_path
        # Simpler resolution: relative to project root
        rel_candidate = os.path.join(
            settings.UPLOAD_DIR, p.photo_path.replace('uploads/', ''))
        candidates = [abs_path, rel_candidate, os.path.join(
            settings.UPLOAD_DIR, p.photo_path)]
        img = None
        for c in candidates:
            if os.path.exists(c):
                img = cv2.imread(c)
                if img is not None:
                    break
        if img is None:
            continue
        emb = _extract_embedding_from_image(img)
        if emb is None or emb.shape[1] != settings.EMBEDDING_DIM:
            continue
        try:
            new_fid = vector_db_instance.add_embedding(student_id, emb)
            # update photo's faiss id
            try:
                p.faiss_vector_id = int(new_fid)
                db.add(p)
            except Exception:
                pass
            readded += 1
        except Exception:
            continue

    student.face_embedding_count = readded
    db.add(student)
    db.commit()
    db.refresh(student)
    return {"readded": readded, "total": student.face_embedding_count}


def delete_student_photo(db: Session, student_id: int, photo_id: int, rebuild: bool = True):
    photo = db.query(models.StudentPhoto).filter(models.StudentPhoto.id ==
                                                 photo_id, models.StudentPhoto.student_id == student_id).first()
    if not photo:
        return {"deleted": False, "message": "Photo not found"}
    # Try fast delete from FAISS by stored vector id
    removed = 0
    if getattr(photo, 'faiss_vector_id', None):
        try:
            removed = vector_db_instance.remove_by_faiss_id(
                photo.faiss_vector_id, student_id)
        except Exception:
            removed = 0
    if removed > 0:
        # decrement student's embedding count
        student = get_student(db, student_id)
        if student:
            student.face_embedding_count = max(
                0, int(student.face_embedding_count or 0) - 1)
            db.add(student)
    # Remove file from disk if exists
    file_path = os.path.join(settings.UPLOAD_DIR, *
                             photo.photo_path.split('/')[1:])
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass
    db.delete(photo)
    db.commit()
    # If fast delete failed and requested, rebuild embeddings to stay consistent
    rebuild_result = None
    if removed == 0 and rebuild:
        rebuild_result = rebuild_student_embeddings(db, student_id)
    return {"deleted": True, "faiss_removed": removed > 0, "rebuild": rebuild_result}


def identify_student_from_image(db: Session, file: UploadFile):
    """Return list of detected faces with matched student info (if any)."""
    if ai_engine.identity_model is None:
        return {"faces": [], "message": "Identity model not loaded"}
    try:
        contents = file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return {"faces": [], "message": "Invalid image"}
        # Normalize channels
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        raw_faces = ai_engine.identity_model.get(img) or []
        results = []
        h, w = img.shape[:2]
        for f in raw_faces:
            bbox = getattr(f, 'bbox', None)
            emb = getattr(f, 'embedding', None)
            if bbox is None or emb is None or len(bbox) < 4:
                continue
            # Prepare embedding
            emb_arr = np.array(emb, dtype='float32')
            if emb_arr.ndim == 1:
                emb_arr = emb_arr.reshape(1, -1)
            elif emb_arr.ndim == 2 and emb_arr.shape[0] != 1:
                emb_arr = emb_arr[0:1]
            if emb_arr.shape[1] != settings.EMBEDDING_DIM:
                continue
            matched_id, similarity = vector_db_instance.search_embedding(
                emb_arr)
            student_data = None
            if matched_id:
                s = get_student(db, matched_id)
                if s:
                    student_data = {
                        "id": s.id,
                        "name": s.name,
                        "student_code": s.student_code,
                        "email": s.email,
                        "class_name": s.class_name,
                        "major": s.major,
                        "status": s.status
                    }
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            results.append({
                "bbox": [x1, y1, x2, y2],
                "similarity": similarity,
                "matched": student_data is not None,
                "student": student_data
            })
        return {"faces": results, "message": "ok"}
    except Exception as e:
        return {"faces": [], "message": str(e)}
