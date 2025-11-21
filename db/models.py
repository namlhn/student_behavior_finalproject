from sqlalchemy import Column, Integer, String, Float, BigInteger, ForeignKey, DateTime, Date, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from core.database import Base


class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True, index=True)
    student_code = Column(String(20), unique=True, index=True)  # Student ID
    name = Column(String(255), index=True)
    email = Column(String(255), unique=True, index=True)
    phone = Column(String(20))
    date_of_birth = Column(Date)
    gender = Column(String(10))  # Male, Female, Other
    address = Column(Text)
    class_name = Column(String(100))
    academic_level = Column(String(50))
    course = Column(String(50))  # Course
    major = Column(String(100))  # Major
    gpa = Column(Float)
    # Status: Active, Inactive, Graduated
    status = Column(String(20), default="Active")
    photo_path = Column(String(512))
    face_embedding_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow,
                        onupdate=datetime.utcnow)
    # No relationships or foreign keys per current design


class StudentPhoto(Base):
    __tablename__ = "student_photos"
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, index=True)
    photo_path = Column(String(512))
    faiss_vector_id = Column(BigInteger, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ClassSession(Base):
    __tablename__ = "class_sessions"
    id = Column(Integer, primary_key=True, index=True)
    # Basic info (text inputs)
    class_name = Column(String(255), index=True)
    subject_name = Column(String(255), index=True)
    teacher_name = Column(String(255))
    student_count = Column(Integer)
    session_date = Column(Date)
    start_time = Column(String(20))
    end_time = Column(String(20))
    room = Column(String(100))
    notes = Column(Text)
    # Video info
    video_path = Column(String(1024))
    # pending, processing, completed, failed
    status = Column(String(50), default="pending")
    duration = Column(Float)
    fps = Column(Float)
    # Aggregated results and details (JSON stored as text)
    behavior_counts = Column(Text)
    emotion_counts = Column(Text)
    per_student_details = Column(Text)
    unknown_counts = Column(Text)
    total_detections = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow,
                        onupdate=datetime.utcnow)

    # PARENT TABLE: Stores behavior events (e.g., a group is Reading)


class SessionBehaviorLog(Base):
    __tablename__ = "session_behavior_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("class_sessions.id"), index=True)

    timestamp = Column(Float, index=True)    # Timestamp (seconds)
    behavior_type = Column(String(50))       # E.g., reading, group_discuss

    # Coordinates of the large behavior bounding box (YOLO Box) on the whole frame
    bbox = Column(String(100))  # "x1,y1,x2,y2"

    # Relationship
    session = relationship("ClassSession")
    # Link to the list of students inside the behavior
    students = relationship(
        "SessionStudentLog", back_populates="behavior_log", cascade="all, delete-orphan")

    # CHILD TABLE: Stores each student in that behavior event


class SessionStudentLog(Base):
    __tablename__ = "session_student_logs"

    id = Column(Integer, primary_key=True, index=True)

    # Backlink to parent behavior log table
    behavior_log_id = Column(Integer, ForeignKey(
        "session_behavior_logs.id"), index=True)

    # Student information
    student_id = Column(Integer, default=0)
    student_name = Column(String(255), default="Unknown")

    # Emotion of THIS student
    emotion = Column(String(50))  # E.g., happy, focused

    # Student's face coordinates (Global coordinates for drawing on video)
    face_bbox = Column(String(100))  # "x1,y1,x2,y2"

    # Relationship
    behavior_log = relationship(
        "SessionBehaviorLog", back_populates="students")
