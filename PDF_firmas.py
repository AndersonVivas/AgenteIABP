import os
import ssl
from typing import List, Dict

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
import cv2


# -------------------------------------------------------------------
# Fix para evitar el error de OpenMP en macOS
# -------------------------------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configuración SSL
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''


# -------------------------------------------------------------------
# OCR de PDF (texto escaneado)
# -------------------------------------------------------------------
def ocr_pdf(file_path: str) -> List[Document]:
    print("🔎 Procesando PDF con OCR (Tesseract)...")
    pages = convert_from_path(file_path, dpi=300)
    docs = []
    for i, page in enumerate(pages):
        txt = pytesseract.image_to_string(
            page,
            lang="spa",
            config="--oem 3 --psm 6"
        )
        txt = " ".join(txt.split())
        docs.append(Document(page_content=txt, metadata={"page": i+1}))
    return docs


# -------------------------------------------------------------------
# Detección de firmas con OpenCV
# -------------------------------------------------------------------
def detectar_firmas(file_path: str) -> Dict[int, bool]:
    print("🖊️ Buscando firmas en el documento...")
    pages = convert_from_path(file_path, dpi=300)
    firmas_por_pagina = {}

    for i, page in enumerate(pages):
        # Guardar imagen temporal
        img_path = f"page_{i+1}.png"
        page.save(img_path, "PNG")

        # Leer en escala de grises
        img = cv2.imread(img_path, 0)
        _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        firmas_detectadas = False
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            # Heurística simple: firmas suelen ser trazos irregulares de tamaño medio
            if 3000 < area < 50000 and h < 150:
                firmas_detectadas = True
                break

        firmas_por_pagina[i+1] = firmas_detectadas
        os.remove(img_path)  # limpiar archivo temporal

    return firmas_por_pagina


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
    print(f"✅ Texto cargado. Páginas: {len(docs)}")

    # Detectar firmas
    firmas = detectar_firmas(pdf_path)

    # Convertir firmas en texto "documento extra"
    firmas_texto = []
    for pagina, hay_firma in firmas.items():
        if hay_firma:
            firmas_texto.append(f"Página {pagina}: Firma detectada")
        else:
            firmas_texto.append(f"Página {pagina}: No se detectó firma")

    firmas_doc = Document(
        page_content="\n".join(firmas_texto),
        metadata={"tipo": "firmas"}
    )
    docs.append(firmas_doc)

    # Dividir en chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"Texto dividido en {len(chunks)} fragmentos.")

    # Embeddings con Ollama (bge-m3)
    embeddings = OllamaEmbeddings(model="bge-m3")

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

    # Prompt mejorado
    prompt = PromptTemplate(
        template="""
        Actúa como un analista legal y financiero especializado en documentos formales.
        Tu tarea es analizar el contexto de un documento y devolver la información de manera clara, precisa y en formato JSON válido.

        Instrucciones:
        1. Responde únicamente con la información que esté explícitamente en el documento (no inventes ni completes).
        2. Si existen elementos repetidos (ejemplo: fechas, montos, nombres), devuélvelos todos en una lista.
        3. Si un dato no está presente, responde con el valor: "No encontrado en el documento".
        4. Identifica el tipo de documento si es posible (ejemplo: contrato, escritura pública, factura, pagaré, certificado).
        5. Extrae toda la información relevante disponible, como:
            - Partes involucradas (personas, empresas, instituciones).
            - Fechas importantes (firma, vigencia, vencimiento, plazos).
            - Montos, valores o cifras económicas.
            - Obligaciones, derechos o condiciones principales.
            - Bienes, servicios o productos mencionados.
            - Firmas y sellos (especifica en qué página se encuentran).
        6. Devuelve SIEMPRE la respuesta en un JSON válido y bien estructurado con el siguiente esquema:
        7. en la respuesta del json omite el \n y los espacios innecesarios.

        {{
            "tipo_documento": "...",
            "partes_involucradas": ["...", "..."],
            "fechas": ["..."],
            "montos": ["..."],
            "obligaciones_condiciones": ["..."],
            "bienes_servicios": ["..."],
            "firmas_sellos": [
                {{
                    "pagina": "...",
                    "detalle": "firma/sello"
                }}
            ],
            "otros_detalles": ["..."]
        }}

        Pregunta: {question}  
        Contexto del documento: {context}  

        Respuesta JSON:
    """,
    input_variables=["question", "context"]
)

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

    # Ejemplo de consultas
    preguntas = [
        "Escanea documento y extrae toda la información relevante en formato JSON.",
    ]

    for pregunta in preguntas:
        print("\n" + "=" * 60)
        print("Pregunta:", pregunta)
        respuesta = qa_chain.invoke(pregunta)
        print("Respuesta del modelo:", respuesta)


if __name__ == "__main__":
    main()
