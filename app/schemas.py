from pydantic import BaseModel, EmailStr
from datetime import date, datetime
from typing import List, Optional


class StudentBase(BaseModel):
    name: str
    email: EmailStr
    student_code: Optional[str] = None
    date_of_birth: Optional[date] = None
    gender: Optional[str] = "Other"
    phone: Optional[str] = None
    address: Optional[str] = None
    class_name: Optional[str] = None
    major: Optional[str] = None
    course: Optional[str] = None
    academic_level: Optional[str] = None
    gpa: Optional[float] = None
    status: Optional[str] = "Active"


class StudentCreate(StudentBase):
    pass


class StudentUpdate(StudentBase):
    # Cho phép các trường có thể thiếu khi update (Partial update)
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    # ... các trường khác cũng optional


class StudentOut(StudentBase):
    id: int
    face_embedding_count: int
    photo_path: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# --- SCHEMAS CHO SESSION & LOGS ---

# 1. Schema cho bảng Con (Student Log)
class StudentLogOut(BaseModel):
    student_id: int
    student_name: str
    emotion: str
    face_bbox: str

    class Config:
        from_attributes = True

# 2. Schema cho bảng Cha (Behavior Log) - Chứa danh sách con


class BehaviorLogOut(BaseModel):
    timestamp: float
    behavior_type: str
    bbox: str
    students: List[StudentLogOut] = []  # Nested list

    class Config:
        from_attributes = True

# 3. Schema cho Session


class SessionBase(BaseModel):
    class_name: str
    subject_name: str
    teacher_name: str
    session_date: date
    start_time: str
    end_time: str
    student_count: Optional[int] = 0
    notes: Optional[str] = None


class SessionCreate(SessionBase):
    pass


class SessionUpdate(BaseModel):
    class_name: Optional[str] = None
    subject_name: Optional[str] = None
    teacher_name: Optional[str] = None
    session_date: Optional[date] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    room: Optional[str] = None
    student_count: Optional[int] = None
    notes: Optional[str] = None


class SessionOut(SessionBase):
    id: int
    video_path: Optional[str] = None
    status: str
    total_detections: int
    created_at: datetime

    class Config:
        from_attributes = True
