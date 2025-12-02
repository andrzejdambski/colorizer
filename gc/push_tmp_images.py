import os
import glob
import logging
from typing import List, Dict
from google.cloud import storage
import json

logger = logging.getLogger(__name__)

def upload_jpgs_to_gcs(
    bucket_name: str | None = None,
    src_dir: str = "tmp_images",
    dest_prefix: str | None = None,
    make_public: bool = False,
) -> List[Dict[str, str]]:
    """
    Upload all .jpg/.jpeg files from src_dir (relative to project root unless absolute)
    to the specified GCS bucket under dest_prefix. Defaults come from env vars:
      - GSC_PROJECT (default: wagon-bootcamp-2130-475009)
      - GSC_FOLDER_PRE_PROC (default: colorizer/pre_proc)

    Returns a list of dicts with local and remote paths.
    """
    # Resolve defaults from environment when None or placeholder strings are provided
    if not bucket_name or bucket_name.startswith("$"):
        bucket_name = os.getenv("GSC_PROJECT", "colorizer")
    if not dest_prefix or dest_prefix.startswith("$"):
        dest_prefix = os.getenv("GSC_FOLDER_PRE_PROC", "preproc")

    # normalize destination prefix so it always ends with a single slash
    dest_prefix = dest_prefix.strip("/") + "/"

    # Resolve project root relative to this file, then the source folder
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_path = src_dir if os.path.isabs(src_dir) else os.path.join(project_root, src_dir)

    if not os.path.isdir(src_path):
        raise FileNotFoundError(f"Source directory not found: {src_path}")

    patterns = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(src_path, pat)))

    if not files:
        logger.info("No JPG/JPEG files found in %s", src_path)
        return []

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    uploaded: List[Dict[str, str]] = []
    for local_path in files:
        filename = os.path.basename(local_path)
        dest = dest_prefix + filename
        blob = bucket.blob(dest)
        blob.content_type = "image/jpeg"
        blob.upload_from_filename(local_path)
        if make_public:
            blob.make_public()
        logger.info("Uploaded %s -> gs://%s/%s", local_path, bucket_name, dest)
        uploaded.append({"local": local_path, "remote": f"gs://{bucket_name}/{dest}"})

    return uploaded


if __name__ == "__main__":
    results = upload_jpgs_to_gcs()
    # print(json.dumps(results, indent=2))
# ...existing code...
