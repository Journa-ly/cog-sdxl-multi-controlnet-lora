import subprocess
import time
import os
import logging
from azure.storage.blob import BlobServiceClient

# Directories and Constants (loaded from environment variables)
AZURE_ACCOUNT_URL = os.getenv("AZURE_ACCOUNT_URL")
JOURNA_CONTAINER_NAME = os.getenv("JOURNA_CONTAINER_NAME")
JOURNA_BLOB_NAME = os.getenv("JOURNA_BLOB_NAME")
JOURNA_MODEL_LOCAL_PATH = os.getenv("JOURNA_MODEL_LOCAL_PATH")
SAS_TOKEN = os.getenv("SAS_TOKEN")

class WeightsDownloader:
    @staticmethod
    def download_if_not_exists(url, dest):
        if not os.path.exists(dest):
            WeightsDownloader.download(url, dest)

    @staticmethod
    def download(url, dest):
        start = time.time()
        print("downloading url: ", url)
        print("downloading to: ", dest)
        subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
        print("downloading took: ", time.time() - start)

    #--Nick

    @staticmethod
    # Function to get BlobServiceClient
    def get_blob_service_client_sas(sas_token: str) -> BlobServiceClient:
        logging.info("Creating BlobServiceClient with SAS token.")
        return BlobServiceClient(account_url=AZURE_ACCOUNT_URL, credential=sas_token)

    @staticmethod
    # Function to download blob to a file
    def download_blob_to_file(blob_service_client: BlobServiceClient, container_name: str, blob_name: str, dest):
        try:
            logging.info(f"Starting download: {container_name}/{blob_name} to {dest}")
            
            # Create blob client
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            
            # Download the blob to the specified file
            with open(dest, "wb") as file:
                download_stream = blob_client.download_blob()
                logging.info(f"Downloading blob {blob_name}.")
                file.write(download_stream.readall())
                logging.info(f"Download completed for blob {blob_name}.")
        except Exception as e:
            logging.error(f"Failed to download blob {blob_name}: {e}")
