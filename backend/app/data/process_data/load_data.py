from langchain_community.document_loaders import PyPDFLoader
import os

def load_pdfs_from_folder(folder_path):
    documents =[]
    #Không tìm thấy folder data.
    if not os.path.exists(folder_path):
        print(f"not found folder: {folder_path}")
        return documents
    files = os.listdir(folder_path)
    pdf_files = [f for f in files if f.endswith(".pdf")]
    
    print(f"Find {len(pdf_files)} file PDF")
    for file in pdf_files:
        file_path = os.path.join(folder_path, file)
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
            print(f"Loader {file} {(len(docs))} page")
        except Exception as e:
            print(f"Error read {file}: {e}")
    return documents
    