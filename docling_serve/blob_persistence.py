"""Azure Blob Storage persistence for docling-serve conversion results."""

import json
import logging
import os
from typing import Optional

_log = logging.getLogger(__name__)

_BLOB_STORAGE_ACCOUNT = os.environ.get("STORAGE_ACCOUNT_NAME", "")
_BLOB_RESULT_CONTAINER = os.environ.get("RESULT_CONTAINER", "")
_BLOB_RESULT_PREFIX = os.environ.get("RESULT_PREFIX", "")


def is_blob_persistence_configured() -> bool:
    return bool(_BLOB_STORAGE_ACCOUNT and _BLOB_RESULT_CONTAINER)


def upload_result_to_blob(task_id: str, result_data: dict) -> None:
    """Upload conversion result JSON to Azure Blob Storage."""
    if not is_blob_persistence_configured():
        return

    try:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient, ContentSettings

        credential = DefaultAzureCredential()
        blob_service = BlobServiceClient(
            account_url=f"https://{_BLOB_STORAGE_ACCOUNT}.blob.core.windows.net",
            credential=credential,
        )
        container_client = blob_service.get_container_client(_BLOB_RESULT_CONTAINER)
        blob_name = f"{_BLOB_RESULT_PREFIX}/{task_id}.json" if _BLOB_RESULT_PREFIX else f"{task_id}.json"
        result_json = json.dumps(result_data, default=str)
        container_client.upload_blob(
            name=blob_name,
            data=result_json,
            overwrite=True,
            content_settings=ContentSettings(content_type="application/json"),
        )
        _log.info(f"Task {task_id} result uploaded to blob {blob_name}")
    except Exception as e:
        _log.error(f"Task {task_id} blob upload failed: {e}")
