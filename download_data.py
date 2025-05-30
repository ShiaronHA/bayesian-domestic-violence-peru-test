import gdown
import os

def download_gdrive_folder(folder_url, relative_output_path):
    """
    Downloads files from a public Google Drive folder to a specified relative path.
    """
    # Construct absolute path from the script's current working directory.
    # This assumes the script is run from the project root.
    project_root = os.getcwd()
    output_path = os.path.join(project_root, relative_output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created directory: {output_path}")
    else:
        print(f"Output directory already exists: {output_path}")

    try:
        print(f"Attempting to download folder contents from {folder_url} into {output_path}...")
        
        # gdown.download_folder will download the contents of the GDrive folder
        # directly into the 'output_path' directory.
        # use_cookies=False is recommended for public folders.
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
    # This is relative to the project root where the script is expected to be run
    target_relative_path = os.path.join("data", "input_data")

    print("Starting data download script...")
    
    # Check if it's likely being run from the project root
    # by looking for a known file from the project root.
    # Adjust 'data_preprocessor.py' if a more stable root indicator file exists.
    if not os.path.exists("data_preprocessor.py"): 
        print("\nWARNING: This script is intended to be run from the root directory of your project")
        print(f"         (e.g., from 'c:\\Users\\shian\\Documents\\tesis\\bayesian-domestic-violence-peru-test\\')")
        print(f"         Current directory: {os.getcwd()}")
        print("         If this is not the project root, the files might be downloaded to an incorrect location.")

    print(f"This script will attempt to download files to: {os.path.join(os.getcwd(), target_relative_path)}")
    download_gdrive_folder(gdrive_folder_url, target_relative_path)
    print("Script finished.")
