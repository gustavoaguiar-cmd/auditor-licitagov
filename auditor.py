import streamlit as st
import os
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Carrega variáveis de ambiente (Local ou Railway)
load_dotenv()

# --- CONFIGURAÇÃO DE SEGURANÇA (LOGIN) ---
# Em um sistema avançado, isso viria de um Banco de Dados.
# Para começar rápido, vamos usar um dicionário simples aqui.
# Formato: "usuario": "senha"
CLIENTES_AUTORIZADOS = {
    "admin": "admin123",        # Você
    "prefeitura_a": "pref2026", # Cliente 1
    "cliente_teste": "123456"   # Cliente 2
}

def check_login():
    """Função simples de verificação de senha"""
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
                st.sidebar.error("Usuário ou senha incorretos.")
        return False
    else:
        st.sidebar.success(f"Logado como: {st.session_state['usuario_atual']}")
        if st.sidebar.button("Sair"):
            st.session_state["logged_in"] = False
            st.rerun()
        return True

# --- FIM CONFIGURAÇÃO LOGIN ---

# Configuração da Página
st.set_page_config(page_title="Lici Auditor - Área Restrita", page_icon="⚖️", layout="wide")

# CSS para visual profissional
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {background-color: #f8f9fa;}
    h1 {color: #0f2c4a;}
</style>
""", unsafe_allow_html=True)

# 1. VERIFICA LOGIN (Bloqueia tudo se não logar)
if not check_login():
    st.title("Lici Auditor ⚖️")
    st.warning("Por favor, faça login na barra lateral para acessar o sistema.")
    st.stop() # Para a execução aqui se não estiver logado

# --- A PARTIR DAQUI, SÓ USUÁRIO LOGADO VÊ ---

# 2. CARREGA A API KEY ESCONDIDA (Do Railway)
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    st.error("ERRO CRÍTICO: API Key não configurada no servidor. Contate o administrador.")
    st.stop()

# Funções Auxiliares
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def get_audit_prompt(doc_type):
    # (Mantendo os mesmos prompts otimizados da v10)
    if doc_type == "Edital de Licitação":
        return """
        Você é um Auditor Especialista em Licitações Públicas no Brasil (Lei 14.133/2021).
        Analise o texto do EDITAL abaixo com rigor extremo.
        
        Sua análise deve ser dividida nas seguintes seções obrigatórias:
        1. ASPECTOS LEGAIS E ESTRUTURAIS (Lei 14.133 citada? Objeto claro? Minuta contrato?)
        2. HABILITAÇÃO E PARTICIPAÇÃO (Busque restrições ilegais, índices financeiros desproporcionais)
        3. REQUISITOS ESSENCIAIS (Prazos, Modo de Disputa, ME/EPP)

        TEXTO DO DOCUMENTO:
        {text}

        SAÍDA ESPERADA:
        Para cada item, diga "CONFORME" ou "NÃO CONFORME".
        Se encontrar cláusula restritiva ou ilegal, inicie a linha com "🚨 ALERTA VERMELHO:".
        Ao final, faça um "RELATÓRIO DE PENDÊNCIAS".
        """
    elif doc_type == "Estudo Técnico Preliminar (ETP)":
        return """
        Auditor da Lei 14.133/21. Analise este ETP com base estrita no Art. 18, §1º.
        Verifique todos os incisos (I ao XIII). Se faltar algo, marque "🚨 ERRO".
        TEXTO: {text}
        """
    elif doc_type == "Termo de Referência (TR)":
        return """
        Auditor da Lei 14.133/21. Analise este TR com base no Art. 6º, XXIII.
        Verifique: Objeto, Fundamentação, Ciclo de vida, Fiscalização, Pagamento, Orçamento.
        TEXTO: {text}
        """
    else: 
        return """Analise este Projeto Básico com base no Art. 6º, XXV da Lei 14.133/21. TEXTO: {text}"""

# Interface Principal
st.title(f"Lici Auditor v11 🏛️")
st.markdown("### Auditoria Jurídica Inteligente - Lei 14.133/21")

# Seleção do Tipo de Documento
doc_type = st.selectbox(
    "Qual documento você vai auditar?",
    ["Edital de Licitação", "Estudo Técnico Preliminar (ETP)", "Termo de Referência (TR)", "Projeto Básico"]
)

# Upload de Arquivo
uploaded_file = st.file_uploader("Faça upload do documento (PDF)", type="pdf")

if uploaded_file and st.button("🔍 Iniciar Auditoria"):
    with st.spinner(f"O Auditor está analisando o documento... (Aguarde alguns segundos)"):
        try:
            # 1. Extrair Texto
            raw_text = get_pdf_text([uploaded_file])
            if len(raw_text) < 50:
                st.error("O PDF parece estar vazio ou é uma imagem digitalizada (scanned). Preciso de texto selecionável.")
            else:
                # 2. Configurar IA com a chave oculta
                llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0, openai_api_key=api_key)
                
                # 3. Executar Análise
                audit_prompt = get_audit_prompt(doc_type)
                prompt = PromptTemplate(template=audit_prompt, input_variables=["text"])
                final_prompt = prompt.format(text=raw_text[:80000]) # Limite seguro
                
                response = llm.invoke(final_prompt)
                
                # 4. Exibir Resultado
                st.success("Auditoria Concluída!")
                st.markdown("### 📋 Relatório de Análise")
                st.markdown(response.content)
                
                st.download_button(
                    label="📥 Baixar Relatório",
                    data=response.content,
                    file_name=f"Auditoria_{doc_type}.md",
                    mime="text/markdown"
                )
        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")
