import ssl
import os
from typing import List
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM

# OCR
from pdf2image import convert_from_path
import pytesseract

# -------------------------------------------------------------------
# Fix para evitar el error de OpenMP en macOS
# -------------------------------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configuración SSL
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''

# -------------------------------------------------------------------
# Función OCR
# -------------------------------------------------------------------
def ocr_pdf(file_path: str) -> List[Document]:
    print("🔎 Aplicando OCR con Tesseract...")
    pages = convert_from_path(file_path, dpi=300)
    docs = []
    for i, page in enumerate(pages):
        text = pytesseract.image_to_string(page, lang="spa")
        docs.append(Document(page_content=text, metadata={"page": i+1}))
    return docs

# -------------------------------------------------------------------
# Cargar PDF con fallback OCR
# -------------------------------------------------------------------
def load_pdf_with_fallback(file_path: str) -> List[Document]:
    print("📄 Intentando extraer texto digital...")
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    if not any(doc.page_content.strip() for doc in docs):
        print("⚠️ PDF parece escaneado, aplicando OCR...")
        docs = ocr_pdf(file_path)

    return docs

# -------------------------------------------------------------------
# Procesar PDF y crear vectorstore
# -------------------------------------------------------------------
def process_pdf(pdf_path: str) -> FAISS:
    docs = load_pdf_with_fallback(pdf_path)
    print(f"✅ PDF procesado. Páginas: {len(docs)}")

    # Dividir en chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=600,chunk_overlap=80,separators=["\n\n", "\n", " ", ""])
    chunks = splitter.split_documents(docs)
    print(f"Texto dividido en {len(chunks)} fragmentos.")

    # Embeddings con Ollama (multilingüe)
    embeddings = OllamaEmbeddings(
        model="bge-m3",              # o "snowflake-arctic-embed2"
    # Estos prefijos ayudan a bge-* a diferenciar consulta vs documento
        embed_instruction="passage: ",
        query_instruction="query: "
    )


    # Crear FAISS vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("✅ Vectorstore creado con embeddings semánticos y OCR (si fue necesario).")
    return vectorstore

# -------------------------------------------------------------------
# Función principal
# -------------------------------------------------------------------
def main():
    pdf_path = "Activos_Fijos.pdf"  # Cambia por tu archivo
    vectorstore = process_pdf(pdf_path)

    # Cargar modelo Ollama
    print("Cargando modelo Ollama...")
    llm = OllamaLLM(model="mistral")
    print("Modelo cargado correctamente ✅")

    # Crear RetrievalQA
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    # Ejemplos de consultas
    preguntas = [
        "¿Cuáles son los montos totales mencionados en el documento?",
        "¿HORA LLEGADA POR EL CLIENTE ENTREGA?",
        "OFICINA REMITENTE?"
    ]

    for pregunta in preguntas:
        print("\n" + "=" * 60)
        print("Pregunta:", pregunta)
        respuesta = qa_chain.invoke(pregunta)
        print("Respuesta del modelo:", respuesta)

if __name__ == "__main__":
    main()
