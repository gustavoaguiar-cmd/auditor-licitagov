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

# --- CONFIGURAÇÃO DE SEGURANÇA (LOGIN) ---
CLIENTES_AUTORIZADOS = {
    "admin": "admin123",        
    "cliente": "solar2025",    
    "teste": "123456"          
}

def show_landing_page():
    """Mostra a tela de boas-vindas antes do login"""
    st.markdown("""
    <style>
        .landing-title { font-size: 3em; color: #0f2c4a; font-weight: bold; text-align: center; margin-top: 50px;}
        .landing-subtitle { font-size: 1.5em; color: #1c4b75; text-align: center; margin-bottom: 30px;}
        .feature-box { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; margin: 10px; text-align: center; flex: 1; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
        .container { display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; }
    </style>
    <div class="landing-title">Lici Auditor 🏛️</div>
    <div class="landing-subtitle">Inteligência Artificial para Controle de Licitações</div>
    
    <div class="container">
        <div class="feature-box">✅ <b>Auditoria Jurídica</b><br>Cruzamento com Lei 14.133/21</div>
        <div class="feature-box">📚 <b>Jurisprudência</b><br>Base de dados do TCU e TCEs</div>
        <div class="feature-box">⚡ <b>Análise de Risco</b><br>Detecção de cláusulas restritivas</div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

def check_login():
    """Gerencia o acesso ao sistema via barra lateral"""
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        show_landing_page() # Mostra a capa bonita
        st.sidebar.title("🔐 Acesso Restrito")
        usuario = st.sidebar.text_input("Usuário")
        senha = st.sidebar.text_input("Senha", type="password")
        
        if st.sidebar.button("Entrar no Sistema"):
            if usuario in CLIENTES_AUTORIZADOS and CLIENTES_AUTORIZADOS[usuario] == senha:
                st.session_state["logged_in"] = True
                st.session_state["usuario_atual"] = usuario
                st.rerun()
            else:
                st.sidebar.error("Credenciais inválidas.")
        return False
    else:
        st.sidebar.success(f"👤 Auditor: {st.session_state['usuario_atual']}")
        if st.sidebar.button("Sair"):
            st.session_state["logged_in"] = False
            st.rerun()
        return True

# --- FUNÇÕES DE INTELIGÊNCIA (RAG OTIMIZADO) ---

@st.cache_resource
def load_knowledge_base():
    """
    Lógica OTIMIZADA: Tenta carregar índice salvo no disco. 
    Se não existir, cria lendo os PDFs e salva para a próxima vez.
    """
    index_path = "faiss_index"
    folder_path = "data/legislacao"
    embeddings = OpenAIEmbeddings()

    # 1. Tenta carregar do disco (Rápido - Cache)
    if os.path.exists(index_path):
        try:
            # allow_dangerous_deserialization é seguro aqui pois nós criamos o arquivo
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            return vectorstore
        except Exception as e:
            print(f"Aviso: Erro ao carregar índice salvo ({e}). Recriando do zero...")

    # 2. Se não existir ou der erro, cria do zero (Lento - Só na 1ª vez ou atualização)
    docs = []
    if not os.path.exists(folder_path):
        return None

    # Varre subpastas recursivamente
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
                        # Salva o nome do arquivo nos metadados para citação
                        docs.append(Document(page_content=text, metadata={"source": filename}))
                except:
                    pass
    
    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # 3. Salva no disco para a próxima vez ser rápida
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

def get_audit_prompt(doc_type):
    # Prompt V13 - Com mais rigor técnico e separação clara
    header = """
    Você é um Auditor de Controle Externo Sênior (perfil rigoroso TCE/ES e TCU).
    Sua missão é cruzar o documento com a Lei 14.133/2021 e a JURISPRUDÊNCIA fornecida.
    Não seja superficial. Aponte o artigo da lei violado ou o Acórdão do TCU ignorado.
    
    Use a jurisprudência fornecida no contexto para embasar suas críticas. Se o edital contraria uma súmula, cite a súmula.

    CONTEXTO JURÍDICO (Use estas fontes):
    {context}

    DOCUMENTO EM ANÁLISE ({doc_type}):
    {text}
    """

    if doc_type == "Edital de Licitação":
        return header + """
        ---
        DIRETRIZES DE ANÁLISE (EDITAL):
        1. HABILITAÇÃO TÉCNICA (Súmula TCU 263):
           - Quantitativos mínimos exigidos ultrapassam 50% do objeto? (Isso é ILEGAL salvo justificativa técnica robusta). Verifique se há justificativa no texto.
           - Certificações (ISO/CMVP): São eliminatórias? Aponte como RISCO se não houver amparo técnico explícito.
        2. HABILITAÇÃO ECONÔMICA (Art. 69, Lei 14.133):
           - Capital Social > 10% do valor estimado? (ILEGAL).
           - Índices Financeiros: São usuais (>1.0)?
        3. MATRIZ DE RISCO:
           - Consta no edital? A ausência é falha grave em serviços continuados ou obras.
        
        GERE RELATÓRIO COM:
        ### 🎯 Resumo Executivo
        ### 🔍 Pente Fino (Cláusulas Restritivas)
        ### ⚖️ Conformidade Legal e Jurisprudencial
        ### 📝 Recomendações
        """
    elif doc_type == "Estudo Técnico Preliminar (ETP)":
        return header + """
        DIRETRIZES (ETP - Art. 18):
        - Houve comparação de soluções de mercado? (Se só indicou uma marca, aponte DIRECIONAMENTO).
        - Justificativa do parcelamento (Súmula 247 TCU) está clara?
        - Há estimativa de valor?
        """
    elif doc_type == "Termo de Referência (TR)":
        return header + """
        DIRETRIZES (TR - Art. 6):
        - Definição do objeto é precisa?
        - Modelo de gestão e fiscalização está definido?
        - Adequação orçamentária foi citada?
        - Critérios de pagamento (medição) estão claros?
        """
    else:
        return header + "Analise o Projeto Básico focando em orçamento detalhado e cronograma físico-financeiro."

# --- INTERFACE PRINCIPAL ---
st.set_page_config(page_title="Lici Auditor v13", page_icon="🏛️", layout="wide")

# CSS Profissional
st.markdown("""
<style>
    .stApp {background-color: #ffffff;}
    h1 {color: #0f2c4a;}
    .stSidebar {background-color: #f0f2f6;}
    div.stButton > button {background-color: #0f2c4a; color: white; border-radius: 5px; border: none;}
    div.stButton > button:hover {background-color: #1c4b75;}
</style>
""", unsafe_allow_html=True)

if not check_login():
    st.stop()

# Área Logada
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ Configuração incompleta: API Key não encontrada no servidor.")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.write("📚 **Status da IA:**")
    # Spinner inteligente: só demora na primeira vez
    with st.spinner("Acessando Cérebro Jurídico..."):
        try:
            vectorstore = load_knowledge_base()
            if vectorstore:
                st.success("✅ Jurisprudência Ativa")
                st.caption("Memória carregada (TCU/TCE)")
            else:
                st.warning("⚠️ Base de dados vazia")
        except Exception as e:
            st.error(f"Erro ao carregar memória: {e}")

st.title("Lici Auditor v13 🏛️")
st.markdown("### Painel de Auditoria - Lei 14.133/21")

col1, col2 = st.columns([1, 2])
with col1:
    doc_type = st.selectbox("Tipo de Documento:", ["Edital de Licitação", "Estudo Técnico Preliminar (ETP)", "Termo de Referência (TR)", "Projeto Básico"])

uploaded_file = st.file_uploader("Faça upload do PDF", type="pdf")

if uploaded_file and st.button("🔍 Iniciar Auditoria"):
    with st.spinner("O Auditor está analisando..."):
        try:
            raw_text = get_pdf_text([uploaded_file])
            if len(raw_text) < 100:
                st.error("PDF sem texto reconhecível (Scanned).")
            else:
                contexto = ""
                if vectorstore:
                    # Busca os 4 trechos mais relevantes na memória
                    docs_rel = vectorstore.similarity_search(raw_text[:4000], k=4)
                    for doc in docs_rel:
                        contexto += f"\n[FONTE: {doc.metadata.get('source','Desconhecida')}]\n...{doc.page_content[:600]}...\n"
                
                # Configura GPT-4 Turbo com temperatura baixa para precisão
                llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.1, openai_api_key=api_key)
                
                prompt_text = get_audit_prompt(doc_type)
                # Passa o doc_type também para o template
                prompt = PromptTemplate(template=prompt_text, input_variables=["context", "text", "doc_type"])
                final_prompt = prompt.format(context=contexto, text=raw_text[:70000], doc_type=doc_type)
                
                response = llm.invoke(final_prompt)
                
                st.success("Análise Finalizada!")
                st.markdown(response.content)
                
                st.download_button("📥 Baixar Relatório", data=response.content, file_name="Auditoria_LiciGov.md")
        except Exception as e:
            st.error(f"Erro: {e}")
