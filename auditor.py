import streamlit as st
import os
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
    "cliente": "solar2025"
}

def check_login():
    st.sidebar.title("🔐 Área do Cliente")
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
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
    else:
        st.sidebar.success(f"Auditor: {st.session_state['usuario_atual']}")
        return True

# --- FUNÇÕES DE INTELIGÊNCIA (RAG) ---

@st.cache_resource
def load_knowledge_base():
    """Lê todos os PDFs da pasta data/legislacao e cria um índice de busca"""
    docs = []
    folder_path = "data/legislacao"
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return None

    # Varre a pasta
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            try:
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    if page.extract_text():
                        text += page.extract_text()
                docs.append(Document(page_content=text, metadata={"source": filename}))
            except:
                pass
    
    if not docs:
        return None

    # Quebra em pedaços menores para busca
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Cria o Banco Vetorial (FAISS)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
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
    return """
    Você é um Auditor de Controle Externo Sênior (perfil TCE/ES e TCU).
    Sua missão não é apenas dizer se está legal, mas encontrar RISCOS e CLÁUSULAS RESTRITIVAS, mesmo que justificadas.

    USE O CONTEXTO ABAIXO (Jurisprudência/Manual) PARA EMBASAR SUA ANÁLISE:
    {context}

    ANÁLISE O DOCUMENTO:
    {text}

    ---
    DIRETRIZES DE AUDITORIA FINA (TCE/ES):
    
    1. HABILITAÇÃO TÉCNICA (O Ponto Crítico):
       - Verifique exigências de quantitativos mínimos (ex: "X kWp", "Y metros"). Calcule mentalmente se parecem ultrapassar 50% do objeto (Súmula TCU 263).
       - Procure siglas como CMVP, PMP, ISO. Se encontrar, VERIFIQUE SE HÁ JUSTIFICATIVA TÉCNICA no texto.
       - Se houver exigência e houver justificativa: Classifique como "⚠️ CLÁUSULA DE RISCO (JUSTIFICADA)". Explique que restringe a competição, mas foi motivado.
       - Se houver exigência SEM justificativa: Classifique como "🚨 ILEGALIDADE (RESTRITIVO)".
    
    2. HABILITAÇÃO ECONÔMICA:
       - Capital Social > 10% do estimado? (Risco à competitividade).
       - Índices de Liquidez > 1.0 (Aceitável) vs > 1.5 (Restritivo).

    3. ORÇAMENTO E BDI:
       - Verifique se menciona BDI diferenciado para materiais/equipamentos (Acórdãos recentes do TCU exigem BDI reduzido para mero fornecimento).
       - Verifique se há matriz de risco.

    ---
    FORMATO DO RELATÓRIO FINAL:

    ### 🎯 1. Resumo Executivo
    (Visão geral da legalidade e risco da licitação)

    ### 🔍 2. Auditoria de Pontos Críticos (Habilitação)
    * **Item Analisado:** (Citar a cláusula, ex: 10.4 - Atestado de 25kWp)
    * **Análise do Auditor:** (Explicar se é proporcional. Citar jurisprudência do contexto se houver).
    * **Veredito:** ✅ REGULAR / ⚠️ RISCO JUSTIFICADO / 🚨 IRREGULAR

    ### 💰 3. Análise Econômica
    (Índices, BDI e Garantias)

    ### ⚖️ 4. Cruzamento com Jurisprudência (Banco de Dados)
    (Cite explicitamente 2 ou 3 decisões/manuais do seu banco de dados que se aplicam a este caso).

    ### 📝 5. Recomendações ao Gestor
    (O que deve ser melhorado ou justificado melhor).
    """

# --- INTERFACE ---
st.set_page_config(page_title="Lici Auditor v12 - Expert", page_icon="⚖️", layout="wide")

# CSS Profissional
st.markdown("""
<style>
    .stApp {background-color: #f8f9fa;}
    h1, h2, h3 {color: #0f2c4a;}
    .stAlert {border-left: 5px solid #ff4b4b;}
</style>
""", unsafe_allow_html=True)

if not check_login():
    st.stop()

# Carrega a "Memória" do Auditor (Jurisprudência)
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("Erro: API Key não configurada no Railway.")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.write("📚 **Base de Conhecimento:**")
    vectorstore = load_knowledge_base()
    if vectorstore:
        st.success("✅ Jurisprudência Carregada (TCU/TCE)")
    else:
        st.warning("⚠️ Nenhuma legislação encontrada em data/legislacao")

st.title("Lici Auditor v12 🏛️ (Expert Mode)")
st.markdown("### Auditoria Fina com Cruzamento de Jurisprudência")

uploaded_file = st.file_uploader("Faça upload do Edital (PDF)", type="pdf")

if uploaded_file and st.button("🔍 Iniciar Auditoria Profunda"):
    with st.spinner("Lendo Edital, consultando Jurisprudência e analisando riscos..."):
        try:
            # 1. Extrair Texto do Edital
            raw_text = get_pdf_text([uploaded_file])
            
            # 2. Buscar Jurisprudência Relevante (RAG)
            contexto_juridico = ""
            if vectorstore:
                # Busca os 4 trechos mais parecidos com o edital no banco de dados
                docs_rel = vectorstore.similarity_search(raw_text[:4000], k=4) 
                for doc in docs_rel:
                    contexto_juridico += f"\n- Fonte: {doc.metadata['source']}\nConteúdo: {doc.page_content[:500]}...\n"
            else:
                contexto_juridico = "Base de conhecimento não carregada. Usar conhecimento geral da Lei 14.133."

            # 3. Analisar com GPT-4
            llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.1, openai_api_key=api_key)
            
            prompt_template = get_audit_prompt("Edital de Licitação")
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "text"])
            
            final_prompt = prompt.format(context=contexto_juridico, text=raw_text[:60000])
            
            response = llm.invoke(final_prompt)
            
            st.markdown(response.content)
            
        except Exception as e:
            st.error(f"Erro na auditoria: {e}")
