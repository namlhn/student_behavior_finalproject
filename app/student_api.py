import random
from core.fastapi_util import AppRouter, api_response_data
from fastapi import Depends, HTTPException, status, UploadFile, File
from core.constants import Result
from sqlalchemy.orm import Session
from typing import List
from core.database import get_db
from core.manager import student_manager
from app import schemas
router = AppRouter()


# 0. Lấy danh sách sinh viên

@router.get("/", response_model=List[dict])
def list_students(search: str = None, db: Session = Depends(get_db)):
    # 1. Lấy danh sách sinh viên từ DB
    students_db = student_manager.get_students(db, search=search, limit=50)

    # 2. "Enrich" dữ liệu
    students_display = []
    for s in students_db:
        s_dict = {
            "id": s.id,
            "name": s.name,
            "student_code": s.student_code,
            "class_name": s.class_name,
            "email": s.email,
            "status": s.status,
            "photo_path": s.photo_path,
            "face_embedding_count": s.face_embedding_count or 0,
            "major": s.major,  # Added missing fields
            "course": s.course,
            "academic_level": s.academic_level,
            "gpa": s.gpa,
            "phone": s.phone,
            "address": s.address,
            "gender": s.gender,
            "date_of_birth": s.date_of_birth
        }

        # --- MOCK DATA FOR DEMO ---
        base_score = random.randint(
            60, 95) if s.face_embedding_count else random.randint(30, 70)
        s_dict['engagement_score'] = base_score
        s_dict['attendance_rate'] = random.randint(70, 100)

        if s_dict['engagement_score'] < 50 or s_dict['attendance_rate'] < 70:
            s_dict['risk_level'] = 'High'
        elif s_dict['engagement_score'] < 70:
            s_dict['risk_level'] = 'Medium'
        else:
            s_dict['risk_level'] = 'Low'

        students_display.append(s_dict)

    return api_response_data(Result.SUCCESS, students_display)


@router.get("/{student_id}", response_model=schemas.StudentOut)
def get_student_detail(student_id: int, db: Session = Depends(get_db)):
    student = student_manager.get_student(db, student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return api_response_data(Result.SUCCESS, student)

# 2. Tạo mới sinh viên


@router.post("/", response_model=schemas.StudentOut, status_code=status.HTTP_201_CREATED)
def create_student(student: schemas.StudentCreate, db: Session = Depends(get_db)):
    # Kiểm tra email trùng
    if student_manager.get_student_by_email(db, student.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    return api_response_data(Result.SUCCESS, student_manager.create_student(db, student))

# 3. Cập nhật sinh viên


@router.put("/{student_id}", response_model=schemas.StudentOut)
def update_student(student_id: int, student: schemas.StudentUpdate, db: Session = Depends(get_db)):
    updated_student = student_manager.update_student(db, student_id, student)
    if not updated_student:
        raise HTTPException(status_code=404, detail="Student not found")
    return api_response_data(Result.SUCCESS, updated_student)


@router.post("/{student_id}/faces")
async def upload_student_faces(
    student_id: int,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    try:
        # Gọi xuống Manager để xử lý logic nghiệp vụ
        result = await student_manager.add_student_faces(db, student_id, files)
        return api_response_data("success", reply=result)
    except Exception as e:
        # Trả lỗi về đúng format
        return api_response_data("error", message=str(e))


@router.get("/{student_id}/photos")
def list_student_photos(student_id: int, db: Session = Depends(get_db)):
    student = student_manager.get_student(db, student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    photos = student_manager.get_student_photos(db, student_id)
    reply = [
        {
            "id": p.id,
            "photo_path": p.photo_path,
            "created_at": p.created_at.isoformat()
        } for p in photos
    ]
    return api_response_data(Result.SUCCESS, reply=reply)


@router.delete("/{student_id}/photos/{photo_id}")
def delete_student_photo(student_id: int, photo_id: int, db: Session = Depends(get_db)):
    result = student_manager.delete_student_photo(
        db, student_id, photo_id, rebuild=True)
    if not result.get("deleted"):
        raise HTTPException(status_code=404, detail=result.get(
            "message", "Delete failed"))
    return api_response_data(Result.SUCCESS, reply=result)


@router.delete("/{student_id}/embeddings")
def delete_all_student_embeddings(student_id: int, db: Session = Depends(get_db)):
    result = student_manager.delete_all_embeddings(db, student_id)
    if result.get("deleted") == 0 and result.get("message"):
        raise HTTPException(status_code=404, detail=result.get("message"))
    # After clearing, optionally rebuild from existing photos (fresh extraction)
    rebuild = student_manager.rebuild_student_embeddings(db, student_id)
    return api_response_data(Result.SUCCESS, reply={"cleared": result, "rebuild": rebuild})


@router.post("/face/recognize")
async def recognize_face(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload an image and attempt to identify student faces."""
    result = student_manager.identify_student_from_image(db, file)
    return api_response_data(Result.SUCCESS, reply=result)
