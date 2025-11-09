import traceback
from pymongo import MongoClient
from gridfs import GridFSBucket
import os
from io import BytesIO
from bson import ObjectId
import hashlib
from pymongo.errors import DuplicateKeyError

class MongoDbTrial:
    def __init__(self):
        self.upload_folder = "temp"
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["qdrant_db"]
        self.fs_bucket = GridFSBucket(self.db)
        self.ensure_dbindex()
        
    def create_folder(self):
        os.makedirs(self.upload_folder, exist_ok=True)
    
    def get_db_details(self):
        return self.db

    def ensure_dbindex(self):
        self.db.fs.files.create_index("metadata.file_hash")
        self.db.extracted_texts.create_index("file_hash",unique=True)
    def compute_file_hash(self,filepath:str):
        hash_256=hashlib.sha256()
        with open(filepath,"rb") as f:
            for chunk in iter(lambda: f.read(4096),b""):
                hash_256.update(chunk)
        return hash_256.hexdigest()


    def save_file(self, files):
        try:
            self.db=self.get_db_details()
            uploaded_files = []
            for filepath in files:
                filename = os.path.basename(filepath)
                file_hash=self.compute_file_hash(filepath=filepath)
                existing=self.db.fs.files.find_one({"metadata.file_hash":file_hash})
                if existing:
                    print(f"Duplicate detected. Skipping {filename}")
                    uploaded_files.append({
                        "filename": filename,
                        "file_id": str(existing["_id"]),
                        "duplicate": True
                    })
                    continue

                with open(filepath, "rb") as f, self.fs_bucket.open_upload_stream(
                    filename,
                    chunk_size_bytes=1048576,
                    metadata={"source": "local_upload","file_hash":file_hash}
                ) as grid_in:
                    while chunk := f.read(1048576):
                        grid_in.write(chunk)
                    file_id=grid_in._id
                uploaded_files.append({"filename":filename,"file_id":str(file_id),"duplicate":False})

            print("Uploaded:", uploaded_files)
            return uploaded_files
        except Exception as e:
            print("Exception occurred in saving file to gridfs",traceback.format_exc())

    def retrieve_files(self, output_folder=None):
        output_folder = output_folder or self.upload_folder
        os.makedirs(output_folder, exist_ok=True)

        for file_doc in self.fs_bucket.find({}):
            output_path = os.path.join(output_folder, file_doc.filename)
            with open(output_path, "wb") as f:
                f.write(file_doc.read())
            print(f"Retrieved: {file_doc.filename}")

    def print_all_res(self, preview_bytes=1000):
        for grid_out in self.fs_bucket.find():
            print(f"\nFilename: {grid_out.filename}")
            if grid_out.filename.endswith(".txt"):
                content = grid_out.read(preview_bytes).decode("utf-8", errors="ignore")
                print("Preview:", content[:preview_bytes], "...")
    
    def retrieve_as_buffers(self):
        file_buffers = {}
        for file_doc in self.fs_bucket.find({}):
            with self.fs_bucket.open_download_stream(file_doc._id) as grid_out:
                file_buffers[file_doc.filename] = grid_out.read()  # bytes
        return file_buffers
    
    def retrieve_by_ids(self,file_ids):
        self.db=self.get_db_details()
        file_buffers={}
        for fid in file_ids:
            if isinstance(fid,str):
                fid=ObjectId(fid)
            try:
                with self.fs_bucket.open_download_stream(fid) as grid_out:
                        file_buffers[grid_out.filename]=BytesIO(grid_out.read())
            except Exception as e:
                print(f"Error retrieving file {fid}: {e}")
        return file_buffers
    
    def check_extracted_text(self,fileids):
        try:
            self.db=self.get_db_details()
            file_buffers={}
            file_meta_data={"file_ids":[],"duplicate_files":[],"file_hash":[]}
            ObjectIds=[ObjectId(fid) if isinstance(fid,str) else fid for fid in fileids]
            file_doc=self.db.fs.files.find({"_id":{"$in":ObjectIds}},{"_id":1,"metadata.file_hash":1})
            file_hash_data={str(doc["_id"]):doc["metadata"]["file_hash"] for doc in file_doc if "metadata" in doc and "file_hash" in doc["metadata"]}
            if not file_hash_data:
                return file_meta_data,file_buffers
            file_hash_data_values=list(file_hash_data.values())
            existing_file_check=self.db.extracted_texts.find({"file_hash":{"$in":file_hash_data_values}},{"file_hash":1,"extracted_text":1})
            existing_hashes={doc["file_hash"] for doc in existing_file_check if doc["extracted_text"]!="" }
            for file_id,file_hash in file_hash_data.items():
                if file_hash not in existing_hashes:
                    with self.fs_bucket.open_download_stream(ObjectId(file_id)) as grid_out:
                        file_buffers[grid_out.filename]=BytesIO(grid_out.read())
                    file_meta_data["file_ids"].append(file_id)
                    file_meta_data["file_hash"].append(file_hash)
                else:
                    file_meta_data["duplicate_files"].append(file_id)
            return file_meta_data,file_buffers
        except Exception as e:
            print("Exception occurred in checking extracted text collection method",traceback.format_exc())
            raise e
 
    def store_extracted_text(self,data):
        data_dict = {
            "filename": data.get("file_name"),
            "extracted_text": data.get("extracted_text"),
            "file_hash": data.get("file_hash"),
            "status": data.get("status")
        }
        try:
            # Use upsert=False to insert only if not existing
            self.db.extracted_texts.insert_one(data_dict)
            return {"status": "inserted", "file_hash": data_dict["file_hash"]}
        except DuplicateKeyError:
            print(f"Duplicate detected for file_hash {data_dict['file_hash']}, skipping insert.")
            return {"status": "duplicate", "file_hash": data_dict["file_hash"]}
        except Exception as e:
            print("Error inserting extracted text:", e)
            return {"status": "error", "error": str(e)}
        
    def statusfilecheck(self,data):
        try:
            data_dict = {
                "filename": data.get("file_name"),
                "status": "success"
            }
            self.db.file_status.insert_one(data_dict)
            return "success"
        except Exception as e:
            print("Insertion unsuccessful",traceback.format_exc())
            return "failure"


