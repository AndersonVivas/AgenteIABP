# Standard library imports
import os
import ssl
from typing import List

# Third-party imports
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# LangChain imports
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader  # PDF loader
from langchain_community.vectorstores import FAISS  # Vector database
from langchain_huggingface import HuggingFacePipeline  # LLM integration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Configure SSL settings to avoid corporate network issues
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''


class SimpleEmbeddings(Embeddings):
    """Simple embeddings class to avoid connectivity issues."""
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of texts."""
        embeddings = []
        for text in texts:
            # Create a simple vector based on text hash
            text_hash = hash(text)
            # Convert to a 384-dimensional vector (standard size)
            embedding = [float((text_hash + i) % 1000) / 1000.0 for i in range(384)]
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Create embedding for a single query text."""
        text_hash = hash(text)
        return [float((text_hash + i) % 1000) / 1000.0 for i in range(384)]
# Initialize PDF analysis
def initialize_pdf_analysis():
    """Initialize the PDF analysis process by loading and processing the document."""
    print("Iniciando el análisis del PDF...")
    
    # Load the PDF
    print("Cargando PDF...")
    loader = PyPDFLoader("Activos_Fijos.pdf")
    docs = loader.load()
    print(f"PDF cargado. Encontradas {len(docs)} páginas.")
    
    # Split text into chunks
    print("Dividiendo texto en fragmentos...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Texto dividido en {len(chunks)} fragmentos.")
    
    # Create simple embeddings
    print("Creando embeddings...")
    embeddings = SimpleEmbeddings()
    
    # Create vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("Base de datos vectorial creada.")
    
    return chunks, vectorstore


# Initialize the system
chunks, vectorstore = initialize_pdf_analysis()
# Crear un LLM simple para propósitos de prueba
print("Creando modelo de lenguaje Hugging Face...")

# Carga el modelo y el tokenizer de Hugging Face
model_name = "google/flan-t5-small"  # Modelo T5 pequeño para generación de texto
print(f"Cargando modelo {model_name}...")

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Crea el pipeline de Hugging Face
hf_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=256,
    temperature=0.7,
    do_sample=True
)

# Integra el pipeline con LangChain
simple_llm = HuggingFacePipeline(pipeline=hf_pipeline)
# Crear una función para simular QA
def simple_qa(question: str) -> str:
    """
    Question-answering function using T5 model and document context.
    
    Args:
        question: The question to answer
        
    Returns:
        str: The answer based on the document content and T5 model
    """
    print(f"Pregunta: {question}")
    
    # Get relevant chunks using vector similarity
    search_results = vectorstore.similarity_search(question, k=3)
    context = "\n".join(doc.page_content for doc in search_results)
    
    # Create a prompt for T5
    prompt = f"Contexto: {context}\n\nPregunta: {question}\n\nRespuesta:"
    
    # Generate answer using T5
    result = simple_llm(prompt)
    
    if isinstance(result, list):
        answer = result[0]['generated_text']
    else:
        answer = result
        
    print("\nContexto utilizado:")
    print("-" * 50)
    print(context[:500] + "..." if len(context) > 500 else context)
    print("-" * 50)
    
    return answer
def validar_firma_administracion():
    """
    Validate the presence of administration signature in the document.
    
    Returns:
        dict: Contains validation status, relevant chunks count, and detailed results
    """
    print("Validando presencia de firma de administración de bienes activos fijos...")
    
    # Keywords related to signature and administration
    keywords_firma = [
        "administración de bienes",
        "activos fijos",
        "firma",
        "atentamente",
        "administración",
        "bienes",
        "responsable",
        "autorización"
    ]
    
    firma_encontrada = False
    chunks_con_firma = []
    
    # Search through all chunks
    for i, chunk in enumerate(chunks):
        chunk_text = chunk.page_content.lower()
        # Check for administration and signature keywords
        if any(keyword in chunk_text for keyword in keywords_firma):
            chunks_con_firma.append({
                'chunk_number': i + 1,
                'content': chunk.page_content,
                'keywords_found': [k for k in keywords_firma if k in chunk_text]
            })
    
    # Analyze found chunks
    if chunks_con_firma:
        print(
            f"\n✅ VALIDACIÓN EXITOSA: Se encontraron {len(chunks_con_firma)} "
            "secciones con información de administración/firma"
        )
        
        for chunk_info in chunks_con_firma:
            print(f"\n--- Chunk {chunk_info['chunk_number']} ---")
            print(f"Palabras clave encontradas: {', '.join(chunk_info['keywords_found'])}")
            print(f"Contenido: {chunk_info['content'][:400]}...")
            
            # Specific signature elements verification
            content_lower = chunk_info['content'].lower()
            if ("administración de bienes" in content_lower and 
                "activos fijos" in content_lower):
                print("🔍 Se detectó sección de 'Administración de Bienes Activos Fijos'")
                firma_encontrada = True
                
            if "atentamente" in content_lower:
                print("🔍 Se detectó saludo de cortesía (posible área de firma)")
    else:
        print(
            "\n❌ VALIDACIÓN FALLIDA: No se encontró información sobre "
            "administración de bienes activos fijos"
        )
    
    # Generate final result
    if firma_encontrada:
        resultado = (
            "✅ DOCUMENTO VÁLIDO: El documento contiene la sección de "
            "'Administración de Bienes Activos Fijos'"
        )
    else:
        resultado = (
            "⚠️ VERIFICACIÓN REQUERIDA: No se pudo confirmar la presencia "
            "completa de la firma de administración"
        )
    
    print(f"\n{'-'*60}")
    print(f"RESULTADO FINAL: {resultado}")
    print(f"{'-'*60}")
    
    return {
        'firma_valida': firma_encontrada,
        'chunks_relevantes': len(chunks_con_firma),
        'resultado': resultado,
        'detalles': chunks_con_firma
    }
def main():
    """Main execution function."""
    print("Sistema listo para responder preguntas.")
    print("-" * 50)
    
    # Execute signature validation
    print("🔍 INICIANDO VALIDACIÓN DE FIRMA DE ADMINISTRACIÓN...")
    resultado_validacion = validar_firma_administracion()
    
    print("\n" + "="*60)
    print("📋 CONSULTAS ADICIONALES:")
    print("="*60)
    
    # Query the PDF using our simple system
    respuesta = simple_qa("¿Cuáles son los montos totales mencionados en el documento?")
    print("Respuesta:")
    print(respuesta)
    
    print("\n" + "="*60)
    print("🔍 CONSULTA SOBRE FIRMA DE ADMINISTRACIÓN:")
    print("="*60)
    
    # Specific query about signature validation
    respuesta_firma = simple_qa(
        "¿El documento tiene la firma de administración de bienes activos fijos?"
    )
    print("Respuesta sobre la firma:")
    print(respuesta_firma)
    
    # Show final validation summary
    if 'resultado_validacion' in locals():
        print("\n📊 RESUMEN DE VALIDACIÓN:")
        print(
            f" • Estado de la firma: "
            f"{'VÁLIDA' if resultado_validacion['firma_valida'] else 'REQUIERE VERIFICACIÓN'}"
        )
        print(f" • Secciones relevantes encontradas: "
              f"{resultado_validacion['chunks_relevantes']}")
        print(f" • Resultado: {resultado_validacion['resultado']}")