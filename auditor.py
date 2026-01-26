import streamlit as st
import os
import time
from pypdf import PdfReader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# --- CONFIGURAÇÃO DE SEGURANÇA ---
CLIENTES_AUTORIZADOS = {
    "admin": "admin123",        
    "cliente": "solar2025",    
    "teste": "123456"          
}

def show_landing_page():
    st.markdown("""
    <style>
        .landing-title { font-size: 3em; color: #0f2c4a; font-weight: bold; text-align: center; margin-top: 50px;}
        .landing-subtitle { font-size: 1.5em; color: #1c4b75; text-align: center; margin-bottom: 30px;}
        .feature-box { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; margin: 10px; text-align: center; flex: 1; }
        .container { display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; }
    </style>
    <div class="landing-title">Lici Auditor v14 🏛️</div>
    <div class="landing-subtitle">Inteligência Artificial Autônoma (Baseada em Manuais TCU)</div>
    <div class="container">
        <div class="feature-box">📚 <b>Conhecimento Profundo</b><br>Analisa baseado nos Manuais e Jurisprudência carregados</div>
        <div class="feature-box">🧠 <b>Raciocínio Jurídico</b><br>Identifica riscos não óbvios</div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

def check_login():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        show_landing_page()
        st.sidebar.title("🔐 Acesso Restrito")
        usuario = st.sidebar.text_input("Usuário")
        senha = st.sidebar.text_input("Senha", type="password")
        if st.sidebar.button("Entrar"):
            if usuario in CLIENTES_AUTORIZADOS and CLIENTES_AUTORIZADOS[usuario] == senha:
                st.session_state["logged_in"] = True
                st.session_state["usuario_atual"] = usuario
                st.rerun()
            else:
                st.sidebar.error("Acesso negado.")
        return False
    return True

# --- MOTOR DE INTELIGÊNCIA (RAG) ---

@st.cache_resource
def load_knowledge_base():
    """
    Carrega a base de conhecimento (Manuais/Decisões).
    Usa cache em disco para performance.
    """
    index_path = "faiss_index"
    folder_path = "data/legislacao"
    embeddings = OpenAIEmbeddings()

    # 1. Tenta carregar índice salvo
    if os.path.exists(index_path):
        try:
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        except:
            pass

    # 2. Se não existir, cria do zero lendo TUDO
    docs = []
    if not os.path.exists(folder_path):
        return None

    # Varredura completa recursiva
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(root, filename)
                try:
                    reader = PdfReader(file_path)
                    text = ""
                    for page in reader.pages:
                        if page.extract_text():
                            text += page.extract_text()
                    if text:
                        docs.append(Document(page_content=text, metadata={"source": filename}))
                except:
                    pass
    
    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def get_autonomous_prompt(doc_type):
    """
    PROMPT AUTÔNOMO:
    Não damos o checklist. Damos a ordem para ele agir como o Manual do TCU.
    """
    return """
    Você é um Auditor Federal de Controle Externo Sênior (Nível TCU).
    
    SUA MISSÃO:
    Realizar uma auditoria profunda ("pente fino") no documento abaixo ({doc_type}), utilizando EXCLUSIVAMENTE a inteligência, os critérios, as súmulas e os entendimentos presentes no CONTEXTO JURÍDICO fornecido (Manuais e Decisões).

    NÃO FAÇA RESUMOS. Aponte falhas, riscos, omissões e irregularidades.
    Se o documento estiver perfeito, duvide e verifique novamente cruzando com a jurisprudência.

    ---
    CONTEXTO JURÍDICO (Sua Base de Conhecimento - Manuais e Decisões):
    {context}
    ---

    DOCUMENTO A SER AUDITADO ({doc_type}):
    {text}

    ---
    DIRETRIZES DE PENSAMENTO (Chain of Thought):
    1. Identifique a natureza do documento ({doc_type}).
    2. Recupere da sua memória (Contexto) quais são os requisitos OBRIGATÓRIOS para este tipo de documento segundo o TCU/TCE.
    3. Cruze cada cláusula do documento com esses requisitos.
    4. Identifique:
       - Restrições indevidas à competitividade.
       - Falta de elementos técnicos essenciais (Projetos, Orçamentos, Cronogramas).
       - Exigências de habilitação abusivas.
       - Direcionamento de marca.
    
    GERE O RELATÓRIO NO SEGUINTE FORMATO:

    ## 🚨 Relatório de Auditoria Autônoma
    
    ### 1. Análise de Legalidade e Conformidade (Cruzamento com Manuais)
    (Para cada falha encontrada, cite: "Conforme o Manual X..." ou "Contrariando a Súmula Y do contexto...")

    ### 2. Pontos de Atenção Crítica (Riscos)
    - **Item Analisado:** [Citar cláusula]
    - **Problema Identificado:** [Explique juridicamente o erro]
    - **Base Legal/Jurisprudencial:** [Cite a fonte do Contexto Jurídico]

    ### 3. Recomendações de Correção
    (O que o gestor deve mudar para evitar apontamento do Tribunal)
    """

# --- INTERFACE ---
st.set_page_config(page_title="Lici Auditor v14 - Autônomo", page_icon="⚖️", layout="wide")

# CSS Limpo
st.markdown("""<style>.stApp {background-color: #ffffff;} h1 {color: #0f2c4a;}</style>""", unsafe_allow_html=True)

if not check_login():
    st.stop()

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ API Key não encontrada.")
    st.stop()

# Carrega Base de Dados
with st.sidebar:
    st.markdown("---")
    st.write("🧠 **Cérebro Jurídico:**")
    with st.spinner("Conectando aos Manuais do TCU..."):
        vectorstore = load_knowledge_base()
        if vectorstore:
            st.success("✅ Base Conectada")
        else:
            st.warning("⚠️ Base Vazia")

st.title("Lici Auditor v14 🏛️")
st.markdown("### Auditoria Autônoma Baseada em Jurisprudência")

col1, col2 = st.columns([1, 2])
with col1:
    doc_type = st.selectbox("Documento:", ["Edital de Licitação", "Estudo Técnico Preliminar (ETP)", "Termo de Referência (TR)", "Projeto Básico"])

uploaded_file = st.file_uploader("Upload do PDF", type="pdf")

if uploaded_file and st.button("🧠 Iniciar Auditoria Profunda"):
    with st.spinner("Lendo Manuais, Cruzando Dados e Auditando..."):
        try:
            raw_text = get_pdf_text([uploaded_file])
            if len(raw_text) < 100:
                st.error("PDF ilegível (Imagem).")
            else:
                contexto = ""
                if vectorstore:
                    # AUMENTADO PARA k=7 para pegar mais contexto dos Manuais
                    docs_rel = vectorstore.similarity_search(raw_text[:6000], k=7)
                    for doc in docs_rel:
                        contexto += f"\n[FONTE: {doc.metadata.get('source','Desconhecida')}]\n...{doc.page_content}...\n"
                
                # Temperature 0.2 para permitir "raciocínio" mas manter fidelidade aos textos
                llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.2, openai_api_key=api_key)
                
                prompt_text = get_autonomous_prompt(doc_type)
                prompt = PromptTemplate(template=prompt_text, input_variables=["context", "text", "doc_type"])
                final_prompt = prompt.format(context=contexto, text=raw_text[:70000], doc_type=doc_type)
                
                response = llm.invoke(final_prompt)
                
                st.success("Auditoria Finalizada")
                st.markdown(response.content)
                st.download_button("📥 Baixar Relatório", data=response.content, file_name="Auditoria_Autonoma.md")
        except Exception as e:
            st.error(f"Erro: {e}")
