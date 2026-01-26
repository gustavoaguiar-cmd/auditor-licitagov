import streamlit as st
import os
from io import BytesIO
from docx import Document as DocxDocument
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
        .landing-subtitle { font-size: 1.2em; color: #555; text-align: center; margin-bottom: 40px;}
        .feature-container { display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin-bottom: 50px;}
        .feature-box { 
            background-color: white; 
            padding: 25px; 
            border-radius: 12px; 
            border: 1px solid #e0e0e0; 
            text-align: center; 
            width: 250px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            transition: transform 0.2s;
        }
        .feature-box:hover { transform: translateY(-5px); }
        .feature-icon { font-size: 2em; margin-bottom: 15px; }
        .feature-text { color: #0f2c4a; font-weight: 600; font-size: 1.1em; }
    </style>
    
    <div class="landing-title">Lici Govtech</div>
    <div class="landing-subtitle">Plataforma de Inteligência em Contratações Públicas</div>
    
    <div class="feature-container">
        <div class="feature-box">
            <div class="feature-icon">🔍</div>
            <div class="feature-text">Auditoria<br>Especializada</div>
        </div>
        <div class="feature-box">
            <div class="feature-icon">⚖️</div>
            <div class="feature-text">Análise Lei<br>14.133/21</div>
        </div>
        <div class="feature-box">
            <div class="feature-icon">🛡️</div>
            <div class="feature-text">Gestão de Riscos<br>e Compliance</div>
        </div>
    </div>
    <hr style="margin-top: 50px; margin-bottom: 50px; opacity: 0.2;">
    """, unsafe_allow_html=True)

def check_login():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        show_landing_page()
        st.sidebar.title("🔐 Login")
        usuario = st.sidebar.text_input("Usuário")
        senha = st.sidebar.text_input("Senha", type="password")
        if st.sidebar.button("Acessar Plataforma"):
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
    """Carrega a base de conhecimento com cache em disco."""
    index_path = "faiss_index"
    folder_path = "data/legislacao"
    embeddings = OpenAIEmbeddings()

    if os.path.exists(index_path):
        try:
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        except:
            pass

    docs = []
    if not os.path.exists(folder_path):
        return None

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

def create_word_docx(markdown_text):
    """Converte o texto da auditoria em um arquivo Word (.docx)"""
    doc = DocxDocument()
    doc.add_heading('Relatório de Auditoria - Lici Govtech', 0)
    
    # Adiciona o texto linha por linha (simples)
    for line in markdown_text.split('\n'):
        if line.startswith('### '):
            doc.add_heading(line.replace('### ', ''), level=2)
        elif line.startswith('## '):
            doc.add_heading(line.replace('## ', ''), level=1)
        elif line.startswith('- ') or line.startswith('* '):
            doc.add_paragraph(line.replace('- ', '').replace('* ', ''), style='List Bullet')
        else:
            doc.add_paragraph(line)
            
    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()

def get_autonomous_prompt(doc_type):
    return """
    Você é um Auditor Federal de Controle Externo Especialista (Nível TCU).
    
    SUA MISSÃO:
    Auditar o documento ({doc_type}) com base na Lei 14.133/21 e na JURISPRUDÊNCIA fornecida.

    REGRAS DE OURO (Siga estritamente):
    1. **LEGISLAÇÃO:** Priorize totalmente a Lei 14.133/2021. Se o contexto trouxer a Lei 8.666/93, IGNORE-A ou mencione apenas se for para dizer que a prática antiga não é mais válida.
    2. **CITAÇÕES:** Ao citar jurisprudência, NÃO cite o nome do arquivo PDF (ex: "manual_tcu_v2.pdf").
       - Em vez disso, PROCURE NO TEXTO DO CONTEXTO o número do Acórdão, Súmula ou Enunciado (ex: "Acórdão 1234/2023 TCU").
       - Se não encontrar o número exato, cite genericamente: "Conforme entendimento consolidado do TCU...".
    3. **RIGOR:** Aponte riscos de sobrepreço, restrição de competitividade e direcionamento.

    ---
    CONTEXTO JURÍDICO (Base de Conhecimento):
    {context}
    ---

    DOCUMENTO A SER AUDITADO ({doc_type}):
    {text}

    GERE O RELATÓRIO NESTE FORMATO:

    ## 🚨 Relatório de Auditoria

    ### 1. Análise de Conformidade (Lei 14.133/21)
    (Análise geral do documento).

    ### 2. Riscos e Irregularidades Identificadas
    - **Ponto Crítico:** [Descreva o problema]
    - **Fundamentação:** [Cite o Acórdão X ou Artigo Y da Lei 14.133 que encontrou no contexto]
    - **Recomendação:** [O que fazer]

    ### 3. Conclusão do Auditor
    """

# --- INTERFACE ---
st.set_page_config(page_title="Lici Govtech", page_icon="🏛️", layout="wide")

# CSS para esconder elementos padrão e deixar limpo
st.markdown("""
<style>
    .stApp {background-color: #ffffff;} 
    h1 {color: #0f2c4a;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

if not check_login():
    st.stop()

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ API Key não encontrada.")
    st.stop()

# Carrega Base de Dados (Silencioso na sidebar)
with st.sidebar:
    st.markdown("---")
    st.write("⚙️ **Sistema:**")
    with st.spinner("Carregando Base Jurídica..."):
        vectorstore = load_knowledge_base()
    if vectorstore:
        st.caption("✅ Base TCU/TCE Ativa")
    else:
        st.caption("⚠️ Base desconectada")

st.title(f"Auditoria: {st.session_state.get('doc_selector', 'Documento')}")

# Menu Superior para Seleção de Documento
# O key='doc_selector' guarda o estado. Usaremos isso para resetar o uploader.
doc_type = st.selectbox(
    "Selecione o tipo de documento para análise:", 
    ["Edital de Licitação", "Estudo Técnico Preliminar (ETP)", "Termo de Referência (TR)", "Projeto Básico"],
    key="doc_selector" 
)

st.info(f"O Auditor aplicará as regras da Lei 14.133/21 específicas para **{doc_type}**.")

# UPLOAD (Com key dinâmica para resetar ao mudar de aba)
uploaded_file = st.file_uploader(
    f"Faça upload do {doc_type} (PDF)", 
    type="pdf", 
    key=f"uploader_{doc_type}" # <--- O PULO DO GATO: Key única por tipo reseta o arquivo
)

if uploaded_file and st.button("🚀 Iniciar Análise"):
    with st.spinner("Analisando cláusulas, cruzando com Acórdãos e gerando parecer..."):
        try:
            raw_text = get_pdf_text([uploaded_file])
            if len(raw_text) < 100:
                st.error("O arquivo PDF parece ser uma imagem. O sistema precisa de texto selecionável.")
            else:
                contexto = ""
                if vectorstore:
                    # Busca inteligência no banco vetorial
                    docs_rel = vectorstore.similarity_search(raw_text[:6000], k=6)
                    for doc in docs_rel:
                        # O prompt instrui a ignorar o 'source' se for nome de arquivo feio
                        contexto += f"\n[TRECHO DA BASE JURÍDICA]:\n{doc.page_content}\n"
                
                llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.2, openai_api_key=api_key)
                
                prompt_text = get_autonomous_prompt(doc_type)
                prompt = PromptTemplate(template=prompt_text, input_variables=["context", "text", "doc_type"])
                final_prompt = prompt.format(context=contexto, text=raw_text[:70000], doc_type=doc_type)
                
                response = llm.invoke(final_prompt)
                
                st.success("Análise Finalizada com Sucesso!")
                st.markdown(response.content)
                
                # Gera arquivo Word
                word_file = create_word_docx(response.content)
                
                st.download_button(
                    label="📄 Baixar Relatório em Word (.docx)",
                    data=word_file,
                    file_name=f"Auditoria_{doc_type.split()[0]}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
        except Exception as e:
            st.error(f"Erro no processamento: {e}")
