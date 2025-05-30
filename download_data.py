import gdown
import os

def download_gdrive_folder(folder_url, relative_output_path):
    """
    Downloads files from a public Google Drive folder to a specified relative path.
    """

    project_root = os.getcwd()
    output_path = os.path.join(project_root, relative_output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created directory: {output_path}")
    else:
        print(f"Output directory already exists: {output_path}")

    try:
        print(f"Attempting to download folder contents from {folder_url} into {output_path}...")
        
        gdown.download_folder(url=folder_url, output=output_path, quiet=False, use_cookies=False)
        
        print(f"Download attempt complete. Please check the directory: {output_path}")
        print("Verify that all expected files are present.")

    except Exception as e:
        print(f"An error occurred during download: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure the Google Drive folder URL is correct and the folder is publicly accessible (set to 'Anyone with the link can view').")
        print("2. Make sure you have 'gdown' installed: pip install gdown")
        print("3. Check your internet connection.")
        print("4. If the folder is very large or contains many files, it might take a while or encounter issues.")
        print("5. If it created an unexpected subfolder inside 'data/input_data', you might need to move the files up one level.")

if __name__ == "__main__":
    gdrive_folder_url = "https://drive.google.com/drive/folders/1Ge8z7mlQg2qGoBehYEhLS8oN5sAhvAQx?usp=drive_link"
    
    target_relative_path = os.path.join("data", "input_data")

    print("Starting data download script...")
    
    if not os.path.exists("data_preprocessor.py"): 
        print(f"         Current directory: {os.getcwd()}")

    print(f"This script will attempt to download files to: {os.path.join(os.getcwd(), target_relative_path)}")
    download_gdrive_folder(gdrive_folder_url, target_relative_path)
    print("Script finished.")
