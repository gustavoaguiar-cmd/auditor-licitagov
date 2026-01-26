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
    """Lê PDFs recursivamente em data/legislacao e TODAS as subpastas"""
    docs = []
    folder_path = "data/legislacao"
    
    if not os.path.exists(folder_path):
        return None

    # CORREÇÃO CRÍTICA: Usa os.walk para entrar nas subpastas (doutrina, tcu, tce...)
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
                    if text: # Só adiciona se extraiu texto
                        # Adiciona metadados com o nome do arquivo para o Auditor citar a fonte
                        docs.append(Document(page_content=text, metadata={"source": filename}))
                except Exception as e:
                    print(f"Erro ao ler {filename}: {e}")
                    pass
    
    if not docs:
        return None

    # Quebra em pedaços para a IA conseguir ler
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Cria o cérebro de busca (Vector Store)
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
    Você é um Auditor de Controle Externo Sênior (perfil rigoroso do TCE/ES e TCU).
    Sua missão é cruzar o Edital com a Lei 14.133 e a JURISPRUDÊNCIA fornecida.

    CONTEXTO JURÍDICO (Use estas fontes para fundamentar):
    {context}

    DOCUMENTO EM ANÁLISE:
    {text}

    ---
    DIRETRIZES DE ANÁLISE PROFUNDA:
    
    1. HABILITAÇÃO TÉCNICA (Foco em Restrição):
       - Atestados: Verifique se a exigência (ex: 25kWp) ultrapassa 50% do objeto. Se o edital justificou, diga "⚠️ RISCO JUSTIFICADO". Se não, "🚨 IRREGULAR (Súmula TCU 263)".
       - Certificações (CMVP, ISO, PMP): São exigíveis? O TCU diz que não podem ser eliminatórias, apenas pontuação técnica (Acórdão 455/2021). Se for eliminatória, marque ERRO.
       
    2. HABILITAÇÃO ECONÔMICA:
       - Capital Social/Patrimônio Líquido: Exigências acima de 10% do valor estimado são ILEGAIS (Lei 14.133, art. 69). Verifique isso.

    3. MATRIZ DE RISCO E MINUTA:
       - A ausência da Matriz de Risco é falha grave em obras/serviços grandes. Aponte.

    ---
    GERE O RELATÓRIO NESTE FORMATO:

    ### 🎯 1. Resumo Executivo
    (Parecer geral sobre a viabilidade jurídica do edital).

    ### 🔍 2. Pente Fino (Cláusulas Polêmicas)
    * **Item Analisado:** (Ex: 10.5 - Exigência de CMVP)
    * **O que o Edital diz:** ...
    * **Jurisprudência Cruzada:** (Aqui você DEVE citar o documento do contexto, ex: "Conforme Informativo TCU nº X...")
    * **Veredito:** ✅ REGULAR / ⚠️ RISCO JUSTIFICADO / 🚨 IRREGULAR

    ### ⚖️ 3. Análise Econômica
    (Índices e Patrimônio Líquido).

    ### 📝 4. Recomendações Corretivas
    (O que mudar para evitar impugnação).
    """

# --- INTERFACE DO SISTEMA ---
st.set_page_config(page_title="Lici Auditor v12 - Expert", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
    .stApp {background-color: #f8f9fa;}
    h1, h2, h3 {color: #0f2c4a;}
    .stAlert {border-left: 5px solid #ff4b4b;}
</style>
""", unsafe_allow_html=True)

if not check_login():
    st.stop()

# Carrega a API Key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("Erro Crítico: API Key não configurada no Railway.")
    st.stop()

# Carrega a Base de Conhecimento (Agora lendo subpastas!)
with st.sidebar:
    st.markdown("---")
    st.write("📚 **Base Jurídica:**")
    with st.spinner("Indexando Manuais e Acórdãos..."):
        vectorstore = load_knowledge_base()
    
    if vectorstore:
        st.success("✅ Biblioteca Jurídica Ativa")
        st.caption("Lendo pastas: legislacao, tcu_informativos, etc.")
    else:
        st.warning("⚠️ Nenhuma legislação encontrada.")

st.title("Lici Auditor v12 🏛️ (Expert Mode)")
st.markdown("### Auditoria com Inteligência Jurisprudencial")

uploaded_file = st.file_uploader("Faça upload do Edital (PDF)", type="pdf")

if uploaded_file and st.button("🔍 Iniciar Auditoria Profunda"):
    with st.spinner("O Auditor está cruzando o Edital com o Banco de Dados..."):
        try:
            # 1. Extrair Texto do Edital
            raw_text = get_pdf_text([uploaded_file])
            
            # 2. Busca Inteligente (RAG)
            contexto_juridico = ""
            if vectorstore:
                # Busca os 5 trechos mais relevantes no seu banco de dados
                docs_rel = vectorstore.similarity_search(raw_text[:4000], k=5) 
                for doc in docs_rel:
                    contexto_juridico += f"\n[FONTE: {doc.metadata['source']}]\n{doc.page_content[:600]}...\n"
            else:
                contexto_juridico = "Sem base jurídica carregada."

            # 3. Análise GPT-4
            llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.1, openai_api_key=api_key)
            
            prompt_template = get_audit_prompt("Edital de Licitação")
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "text"])
            
            # Monta o prompt final com o contexto recuperado das pastas
            final_prompt = prompt.format(context=contexto_juridico, text=raw_text[:60000])
            
            response = llm.invoke(final_prompt)
            
            # 4. Exibe Resultado
            st.markdown(response.content)
            
        except Exception as e:
            st.error(f"Erro na auditoria: {e}")
