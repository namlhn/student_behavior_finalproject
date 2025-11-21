import os
import json
import faiss
import numpy as np
from core.config import settings


class VectorDB:
    def __init__(self):
        self.dim = settings.EMBEDDING_DIM
        self.index_file = settings.FAISS_INDEX_FILE
        self.metadata_file = settings.METADATA_FILE

        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
        # metadata structure (v2):
        # {
        #   "counts": {"<student_id>": int, ...},
        #   "by_student": {"<student_id>": [<faiss_id>, ...]},
        #   "last_id": <int>
        # }
        self.metadata = {}

        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            print("Loading existing FAISS index...")
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                try:
                    self.metadata = json.load(f)
                except Exception:
                    self.metadata = {}
            # Migrate legacy metadata (v1 -> v2)
            if not isinstance(self.metadata, dict):
                self.metadata = {}
            if 'counts' not in self.metadata or 'by_student' not in self.metadata:
                legacy = self.metadata if isinstance(
                    self.metadata, dict) else {}
                counts = {}
                for k, v in legacy.items():
                    try:
                        # keep only student-id integer keys
                        int(k)
                        counts[k] = int(v)
                    except Exception:
                        continue
                self.metadata = {
                    'counts': counts,
                    'by_student': {},
                    'last_id': int(__import__('time').time() * 1000)
                }
                self._save()
        else:
            print("Creating new FAISS index...")
            self._ensure_parent_dirs()
            # initialize fresh metadata v2 structure
            self.metadata = {'counts': {}, 'by_student': {},
                             'last_id': int(__import__('time').time() * 1000)}
            self._save()

    def _ensure_parent_dirs(self):
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)

    def _save(self):
        self._ensure_parent_dirs()
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f)

    def _next_faiss_id(self) -> int:
        # monotonically increasing unique int64 id
        last = int(self.metadata.get('last_id', 0) or 0)
        new_id = last + 1
        self.metadata['last_id'] = new_id
        return int(new_id)

    def add_embedding(self, student_id: int, vector: np.ndarray) -> int:
        """Add embedding and return the FAISS vector id used."""
        vec = vector.astype('float32')
        faiss.normalize_L2(vec)
        faiss_id_int = self._next_faiss_id()
        faiss_id = np.array([faiss_id_int], dtype=np.int64)
        self.index.add_with_ids(vec, faiss_id)
        sid = str(student_id)
        # update counts
        counts = self.metadata.setdefault('counts', {})
        counts[sid] = int(counts.get(sid, 0) or 0) + 1
        # map faiss ids per student
        by_stu = self.metadata.setdefault('by_student', {})
        ids_list = by_stu.get(sid, [])
        ids_list.append(faiss_id_int)
        by_stu[sid] = ids_list
        self._save()
        print(
            f"Added embedding for student {student_id} (faiss_id={faiss_id_int}). Total vectors: {self.index.ntotal}")
        return faiss_id_int

    def search_embedding(self, vector: np.ndarray, k: int = 1):
        if self.index.ntotal == 0:
            return None, 0.0
        vec = vector.astype('float32')
        faiss.normalize_L2(vec)
        distances, faiss_ids = self.index.search(vec, k)
        if faiss_ids.size == 0:
            return None, 0.0
        student_id = int(faiss_ids[0][0])
        similarity = float(distances[0][0])
        if similarity < settings.FAISS_THRESHOLD_COSINE:
            return None, similarity
        # If we are on v2 ids (per-photo id), need to resolve to student via metadata
        by_stu = self.metadata.get('by_student') or {}
        if by_stu:
            # reverse lookup: find student whose list contains the faiss id
            fid = int(faiss_ids[0][0])
            for sid, id_list in by_stu.items():
                if isinstance(id_list, list) and fid in id_list:
                    try:
                        return int(sid), similarity
                    except Exception:
                        break
            # fallback to treat id as student_id if not found
        return student_id, similarity

    def get_embedding_count(self, student_id: int) -> int:
        sid = str(student_id)
        counts = self.metadata.get('counts', {})
        return int(counts.get(sid, 0) or 0)

    def delete_embeddings_for_student(self, student_id: int) -> int:
        sid = str(student_id)
        removed_total = 0
        by_stu = self.metadata.get('by_student', {})
        ids = by_stu.get(sid, [])
        try:
            if ids:
                arr = np.array(ids, dtype=np.int64)
                removed_total = int(self.index.remove_ids(arr))
            else:
                # legacy fallback: vectors were added with id == student_id
                removed_total = int(self.index.remove_ids(
                    np.array([student_id], dtype=np.int64)))
        except Exception:
            removed_total = 0

        # reset metadata for student
        counts = self.metadata.setdefault('counts', {})
        counts[sid] = 0
        if sid in by_stu:
            by_stu[sid] = []
        self._save()
        return removed_total

    def remove_by_faiss_id(self, faiss_id: int, student_id: int | None = None) -> int:
        """Remove a single vector by its FAISS id. Optionally update metadata with student_id.
        Returns number of removed vectors (0 or 1)."""
        try:
            removed = int(self.index.remove_ids(
                np.array([int(faiss_id)], dtype=np.int64)))
        except Exception:
            removed = 0
        if removed > 0:
            # update metadata bookkeeping
            by_stu = self.metadata.setdefault('by_student', {})
            counts = self.metadata.setdefault('counts', {})
            if student_id is not None:
                sid = str(student_id)
                if sid in by_stu and isinstance(by_stu[sid], list):
                    try:
                        by_stu[sid].remove(int(faiss_id))
                    except ValueError:
                        pass
                counts[sid] = max(0, int(counts.get(sid, 0) or 0) - 1)
            else:
                # best-effort reverse lookup
                for sid, id_list in by_stu.items():
                    if isinstance(id_list, list) and int(faiss_id) in id_list:
                        try:
                            id_list.remove(int(faiss_id))
                        except ValueError:
                            pass
                        counts[sid] = max(0, int(counts.get(sid, 0) or 0) - 1)
                        break
            self._save()
        return removed


vector_db_instance = VectorDB()
