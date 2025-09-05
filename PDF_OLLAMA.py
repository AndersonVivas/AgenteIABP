import ssl
import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.prompts import PromptTemplate

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
    print("🔎 Procesando PDF con OCR (Tesseract)...")
    pages = convert_from_path(file_path, dpi=300)
    docs = []
    for i, page in enumerate(pages):
        txt = pytesseract.image_to_string(
            page,
            lang="spa",
            config="--oem 3 --psm 6"   # modo bloque, mejor para formularios
        )
        txt = " ".join(txt.split())    # normalizar espacios
        docs.append(Document(page_content=txt, metadata={"page": i+1}))
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
    docs = ocr_pdf(pdf_path)
    print(f"✅ OCR completado. Páginas: {len(docs)}")

    # Dividir en chunks más pequeños
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"Texto dividido en {len(chunks)} fragmentos.")

    # Embeddings con Ollama (bge-m3)
    embeddings = OllamaEmbeddings(
        model="bge-m3"
    )

    # Crear FAISS vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("✅ Vectorstore creado con embeddings bge-m3.")
    return vectorstore

# -------------------------------------------------------------------
# Función principal
# -------------------------------------------------------------------
def main():
    pdf_path = "Activos_Fijos.pdf"  # Cambia por tu archivo
    vectorstore = process_pdf(pdf_path)

    # Cargar modelo Ollama (Mistral)
    print("Cargando modelo Ollama (Mistral)...")
    llm = OllamaLLM(model="mistral")
    print("Modelo cargado correctamente ✅")

    # Prompt anti-alucinación
    prompt = PromptTemplate(
        template="""
            Actúa como un analista legal y financiero. 
            Tu tarea es leer el contexto de un documento y devolver información de forma clara y exacta.

            Instrucciones:
            - Responde solo con lo que esté en el documento (no inventes).
            - Si se repite información (ej: fechas, montos), devuelve todas en una lista.
            - Si no se encuentra, responde: "No encontrado en el documento".
            - Identifica el tipo de documento si es posible (ej: contrato, escritura, factura).
            - Si el documento tiene firmas, fechas, montos, nombres de partes o cláusulas importantes, devuélvelos claramente.
            - Mantén la redacción concisa y profesional.

            Pregunta: {question}  
            Contexto: {context}  

            Respuesta:
            """,
        input_variables=["question", "context"]
    )

    # Crear RetrievalQA con MMR y más cobertura
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 40, "lambda_mult": 0.3}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    # Ejemplos de consultas
    preguntas = [
        "¿Qué tipo de documento es este?",
        "¿Quiénes son las partes involucradas?",
        "¿Cuáles son los montos o valores mencionados?",
        "¿Tiene firmas y de quién?",
        "¿Qué fechas importantes se mencionan?",
        "¿Existen cláusulas de rescisión o condiciones especiales?"
    ]


    for pregunta in preguntas:
        print("\n" + "=" * 60)
        print("Pregunta:", pregunta)
        respuesta = qa_chain.invoke(pregunta)
        print("Respuesta del modelo:", respuesta)

if __name__ == "__main__":
    main()
