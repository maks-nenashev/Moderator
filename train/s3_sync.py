import os
import sys
from pathlib import Path
import boto3
from dotenv import load_dotenv

load_dotenv()

def get_s3_client():
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region = os.getenv('AWS_DEFAULT_REGION', 'eu-central-1')

    if not access_key or not secret_key:
        raise ValueError("Critical Error: AWS credentials not found in environment variables.")

    return boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )

def upload_version_artifacts(version: str):
    """Загружает артефакты указанной версии из artifacts/<version> в S3."""
    s3 = get_s3_client()
    bucket = os.getenv('S3_BUCKET_NAME', 'findway-ml-artifacts')
    artifact_dir = Path(f"artifacts/{version}")

    if not artifact_dir.exists():
        print(f"❌ Upload Error: Directory {artifact_dir} does not exist.")
        sys.exit(1)

    required_files = ['model.joblib', 'vectorizer.joblib', 'thresholds.json']
    
    print(f"📦 Uploading artifacts for version [{version}] to s3://{bucket}/models/{version}/...")

    for file_name in required_files:
        file_path = artifact_dir / file_name
        if not file_path.exists():
            print(f"⚠️ Warning: File {file_name} not found in {artifact_dir}, skipping.")
            continue

        s3_key = f"models/{version}/{file_name}"
        s3.upload_file(str(file_path), bucket, s3_key)
        print(f"  └─ Uploaded: {file_name} -> {s3_key}")

    print(f"✅ Version [{version}] successfully synced with S3.\n")

def download_version_artifacts(version: str) -> bool:
    """Скачивает артефакты указанной версии из S3 в artifacts/<version>."""
    s3 = get_s3_client()
    bucket = os.getenv('S3_BUCKET_NAME', 'findway-ml-artifacts')
    artifact_dir = Path(f"artifacts/{version}")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    required_files = ['model.joblib', 'vectorizer.joblib', 'thresholds.json']
    
    print(f"📥 Downloading artifacts for version [{version}] from s3://{bucket}/models/{version}/...")

    try:
        for file_name in required_files:
            s3_key = f"models/{version}/{file_name}"
            target_path = artifact_dir / file_name
            s3.download_file(bucket, s3_key, str(target_path))
            print(f"  └─ Downloaded: {file_name}")
        print(f"✅ Version [{version}] successfully downloaded from S3.\n")
        return True
    except Exception as e:
        print(f"❌ Failed to download artifacts for [{version}] from S3: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python s3_sync.py [upload|download] <version>")
        sys.exit(1)

    action = sys.argv[1]
    ver = sys.argv[2]

    if action == "upload":
        upload_version_artifacts(ver)
    elif action == "download":
        download_version_artifacts(ver)
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)