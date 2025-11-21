import sys
import os
import argparse
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

# Add project root to path
sys.path.append(os.getcwd())

try:
    from core.config import settings
    print(f"DATABASE_URL: {settings.DATABASE_URL}")

    if not settings.DATABASE_URL:
        print("Error: DATABASE_URL is empty. model_post_init might not have run.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Verify DB and run minor migrations")
    parser.add_argument("--migrate-faiss-id", action="store_true",
                        help="Ensure student_photos.faiss_vector_id is BIGINT")
    args = parser.parse_args()

    engine = create_engine(settings.DATABASE_URL)
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        print("Database connection successful:", result.fetchone())

        if args.migrate_faiss_id:
            print("Checking column type for student_photos.faiss_vector_id ...")
            q = text(
                """
                SELECT DATA_TYPE 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = DATABASE() 
                  AND TABLE_NAME = 'student_photos' 
                  AND COLUMN_NAME = 'faiss_vector_id'
                """
            )
            dtype_row = connection.execute(q).fetchone()
            current = dtype_row[0] if dtype_row else None
            print(f"Current DATA_TYPE: {current}")
            if current is None:
                print("Column faiss_vector_id not found; creating BIGINT NULL column...")
                connection.execute(
                    text("ALTER TABLE student_photos ADD COLUMN faiss_vector_id BIGINT NULL"))
                print("Added faiss_vector_id as BIGINT.")
            elif current.lower() != 'bigint':
                print("Altering faiss_vector_id to BIGINT ...")
                connection.execute(
                    text("ALTER TABLE student_photos MODIFY COLUMN faiss_vector_id BIGINT NULL"))
                print("Altered faiss_vector_id to BIGINT.")
            else:
                print("faiss_vector_id already BIGINT. No change.")
except Exception as e:
    print(f"Connection failed: {e}")
    sys.exit(1)
