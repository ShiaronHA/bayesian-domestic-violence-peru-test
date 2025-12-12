import os
import logging
import gdown

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_gdrive_folder(folder_url: str, relative_output_path: str) -> None:
    """
    Downloads files from a public Google Drive folder to a specified relative path.
    
    Args:
        folder_url (str): The public URL of the Google Drive folder.
        relative_output_path (str): The relative path where files should be downloaded.
    """
    project_root = os.getcwd()
    output_path = os.path.join(project_root, relative_output_path)

    if not os.path.exists(output_path):
        try:
            os.makedirs(output_path)
            logger.info(f"Created directory: {output_path}")
        except OSError as e:
            logger.error(f"Failed to create directory {output_path}: {e}")
            return
    else:
        logger.info(f"Output directory already exists: {output_path}")

    try:
        logger.info(f"Attempting to download folder contents from {folder_url} into {output_path}...")
        
        # gdown.download_folder handles the download logic
        files = gdown.download_folder(url=folder_url, output=output_path, quiet=False, use_cookies=False)
        
        if files:
            logger.info(f"Download complete. {len(files) if files else 0} files downloaded to: {output_path}")
        else:
            logger.warning("Download completed but no files were returned. Check if the folder is empty or permissions are correct.")

    except Exception as e:
        logger.error(f"An error occurred during download: {e}")
        logger.info("Troubleshooting tips:")
        logger.info("1. Ensure the Google Drive folder URL is correct and publicly accessible.")
        logger.info("2. Check your internet connection.")
        logger.info("3. Verify 'gdown' is installed correctly.")

def main():
    """
    Main execution function.
    """
    gdrive_folder_url = "https://drive.google.com/drive/folders/1Ge8z7mlQg2qGoBehYEhLS8oN5sAhvAQx?usp=drive_link"
    target_relative_path = os.path.join("data", "input_data")

    logger.info("Starting data download script...")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    download_gdrive_folder(gdrive_folder_url, target_relative_path)
    
    logger.info("Script finished.")

if __name__ == "__main__":
    main()
