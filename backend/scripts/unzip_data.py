import zipfile
import os
from tqdm import tqdm

ZIP_PATH = r"C:\Users\Aishik\Documents\Workshop\Algai\Data\DataBento S&P Future (3m).zip"
EXTRACT_DIR = r"C:\Users\Aishik\Documents\Workshop\Algai\Data\Extracted"

def unzip_dataset():
    if not os.path.exists(EXTRACT_DIR):
        os.makedirs(EXTRACT_DIR)
        
    print(f"Opening Zip: {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        total_files = len(zip_ref.infolist())
        print(f"Extracting {total_files} files to {EXTRACT_DIR}...")
        
        for member in tqdm(zip_ref.infolist(), desc="Extracting"):
            try:
                zip_ref.extract(member, EXTRACT_DIR)
            except Exception as e:
                print(f"Error extracting {member.filename}: {e}")
                
    print("Extraction Complete.")

if __name__ == "__main__":
    unzip_dataset()
