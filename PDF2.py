import ssl
import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader  # carga del pdf
from langchain_community.vectorstores import FAISS  # base de datos vectorial
from langchain.embeddings.base import Embeddings
from langchain.chains import RetrievalQA  # patrón pregunta-respuesta
from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline  # ✅ nuevo import

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


def main():
    print("Iniciando el análisis del PDF...")

    # -------------------------------------------------------------------
    # 1. Cargar PDF
    # -------------------------------------------------------------------
    print("Cargando PDF...")
    loader = PyPDFLoader("Activos_Fijos.pdf")  # cambia por tu PDF
    docs = loader.load()
    print(f"PDF cargado. Encontradas {len(docs)} páginas.")

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
        "¿El documento tiene la firma de administración de bienes activos fijos?"
    ]

    for pregunta in preguntas:
        print("\n" + "=" * 60)
        print("Pregunta:", pregunta)
        respuesta = qa_chain.invoke(pregunta)  # ✅ ahora usa invoke
        print("Respuesta del modelo:", respuesta)


if __name__ == "__main__":
    main()
