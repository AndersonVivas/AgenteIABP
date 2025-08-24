import ssl
import os
from typing import List
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader  # carga del pdf
from langchain_community.vectorstores import FAISS  # base de datos vectorial
from langchain.embeddings.base import Embeddings
from langchain.chains import RetrievalQA  # patrón pregunta-respuesta
from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline  # Updated import path

# OCR
from pdf2image import convert_from_path
import pytesseract
from langchain.schema import Document

# -------------------------------------------------------------------
# Fix para evitar el error de OpenMP en macOS
# -------------------------------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configuración de SSL (para evitar problemas en redes corporativas)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''


# -------------------------------------------------------------------
# Clase simple de embeddings (puedes cambiarla por HuggingFaceEmbeddings)
# -------------------------------------------------------------------
class SimpleEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            text_hash = hash(text)
            embedding = [float((text_hash + i) % 1000) / 1000.0 for i in range(384)]
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        text_hash = hash(text)
        return [float((text_hash + i) % 1000) / 1000.0 for i in range(384)]


# -------------------------------------------------------------------
# Función auxiliar: OCR para PDFs escaneados
# -------------------------------------------------------------------
def ocr_pdf(file_path: str):
    print("🔎 Usando OCR con Tesseract...")
    pages = convert_from_path(file_path, dpi=300)  # convertir PDF a imágenes
    docs = []
    for i, page in enumerate(pages):
        text = pytesseract.image_to_string(page, lang="spa")  # OCR en español
        docs.append(Document(page_content=text, metadata={"page": i + 1}))
    return docs


def load_pdf_with_fallback(file_path: str):
    """Carga un PDF con PyPDFLoader, si no tiene texto usa OCR."""
    print("📄 Intentando extraer texto digital...")
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Si todas las páginas están vacías -> usar OCR
    if not any(doc.page_content.strip() for doc in docs):
        print("⚠️ PDF parece escaneado, aplicando OCR...")
        docs = ocr_pdf(file_path)

    return docs


def main():
    print("Iniciando el análisis del PDF...")

    # -------------------------------------------------------------------
    # 1. Cargar PDF (texto digital u OCR)
    # -------------------------------------------------------------------
    pdf_path = "Activos_Fijos.pdf"  # cambia por tu archivo
    docs = load_pdf_with_fallback(pdf_path)
    print(f"PDF procesado. Encontradas {len(docs)} páginas.")

    # -------------------------------------------------------------------
    # 2. Dividir texto en chunks
    # -------------------------------------------------------------------
    print("Dividiendo texto en fragmentos...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Texto dividido en {len(chunks)} fragmentos.")

    # -------------------------------------------------------------------
    # 3. Crear embeddings y vectorstore
    # -------------------------------------------------------------------
    print("Creando embeddings y base vectorial...")
    embeddings = SimpleEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("Base de datos vectorial creada.")

    # -------------------------------------------------------------------
    # 4. Cargar modelo de Hugging Face
    # -------------------------------------------------------------------
    print("Cargando modelo de Hugging Face...")
    model_name = "google/flan-t5-base"  # puedes cambiarlo por otro modelo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    hf_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        repetition_penalty=1.2
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    print("Modelo cargado correctamente ✅")

    # -------------------------------------------------------------------
    # 5. Crear RetrievalQA
    # -------------------------------------------------------------------
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    # -------------------------------------------------------------------
    # 6. Ejemplos de consultas
    # -------------------------------------------------------------------
    preguntas = [
        "¿Cuáles son los montos totales mencionados en el documento?",
        "¿cual es la HORA LLEGADA por el cliente que entrega?"
    ]

    for pregunta in preguntas:
        print("\n" + "=" * 60)
        print("Pregunta:", pregunta)
        respuesta = qa_chain.invoke(pregunta)
        print("Respuesta del modelo:", respuesta)


if __name__ == "__main__":
    main()
