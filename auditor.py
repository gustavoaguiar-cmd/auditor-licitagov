import streamlit as st
import os
import time
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="AUDITOR LICI TECHGOV", page_icon="⚖️", layout="wide")

# --- CSS VISUAL ---
st.markdown("""
<style>
.alert-box {
    background-color: #ffe6e6;
    border-left: 6px solid #ff4b4b;
    padding: 15px;
    margin-bottom: 20px;
    border-radius: 5px;
    color: #333;
}
.success-box {
    background-color: #e6fffa;
    border-left: 6px solid #00cc99;
    padding: 15px;
    margin-bottom: 20px;
    border-radius: 5px;
    color: #333;
}
.neutral-box {
    background-color: #f0f2f6;
    border-left: 6px solid #555;
    padding: 15px;
    margin-bottom: 20px;
    border-radius: 5px;
    color: #333;
}
</style>
""", unsafe_allow_html=True)

# --- SESSÃO ---
if 'logged' not in st.session_state: st.session_state['logged'] = False

# --- 1. LOGIN ---
def check_login(key):
    users = {
        "AMIGO_TESTE": 3,
        "PREFEITURA_X": 10,
        "GUSTAVO_ADMIN": 999
    }
    return users.get(key, -1)

# --- 2. CARREGAMENTO DA BASE ---
@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    text = ""
    data_folder = "data"
    
    if not os.path.exists(data_folder):
        return None, ["ERRO CRÍTICO: Pasta 'data' não encontrada."]

    files_log = []
    for root, dirs, files in os.walk(data_folder):
        for filename in files:
            if filename.lower().endswith('.pdf'):
                filepath = os.path.join(root, filename)
                try:
                    pdf_reader = PdfReader(filepath)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            clean_page = page_text.replace('\x00', '')
                            text += f"\n[FONTE JURÍDICA: {filename}] {clean_page}"
                    files_log.append(f"✅ Base Carregada: {filename}")
                except Exception:
                    files_log.append(f"❌ Erro ao ler base: {filename}")
                    continue
    
    if text == "": return None, files_log

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\nArt.", "\n\n", ". ", " ", ""]
    )
    chunks_raw = text_splitter.split_text(text)
    chunks = [c for c in chunks_raw if c and len(c.strip()) > 20] 
    
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key: return None, ["ERRO: Chave API ausente."]
    
    # Chunk size menor para envio de embeddings (evita timeout no upload)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key, chunk_size=100)
    
    try:
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
        return vectorstore, files_log
    except Exception as e:
        return None, [f"ERRO CRÍTICO OPENAI: {str(e)}"]

# --- 3. CÉREBRO JURÍDICO (PROMPT) ---
def get_audit_chain():
    
    prompt_template = """
    Você é um Auditor Sênior Especialista em Licitações Públicas (Lei 14.133/21).
    
    INSTRUÇÃO DE VARREDURA (Buscando Erros):
    1. LEIA O TEXTO INTEIRO fornecido.
    2. Identifique TODAS as irregularidades, restrições indevidas, omissões obrigatórias ou cláusulas vagas.
    3. Cruze com a Jurisprudência fornecida.
    
    TEMA DA ANÁLISE: {question}
    
    CONTEXTO JURÍDICO:
    {context}
    
    PARECER DO AUDITOR:
    - Se achar erro/restrição: Comece com "🚨 ALERTA".
    - Se faltar item obrigatório: Comece com "⚠️ OMISSÃO".
    - Se estiver tudo certo: Comece com "✅ CONFORME" e cite onde achou.
    - Seja extremamente técnico e cite os artigos.
    """

    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    # --- MUDANÇA ESTRATÉGICA: GPT-4o-MINI ---
    # Motivo: Aguenta 128k tokens mas tem limite de TPM muito maior que o 4o standard.
    # Isso resolve o erro 429 para documentos gigantes.
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# --- 4. MOTOR DE AUDITORIA (PROCESSAMENTO) ---
def process_audit_full(vectorstore, uploaded_file, audit_protocol):
    reader = PdfReader(uploaded_file)
    doc_text = ""
    
    # Extração
    for i, page in enumerate(reader.pages):
        content = page.extract_text()
        if content:
            doc_text += f"\n--- PÁGINA {i+1} ---\n{content.replace(chr(0), '')}"
    
    # Verifica tamanho
    if len(doc_text) < 50:
        return [("Erro", "Arquivo vazio.")]

    chain = get_audit_chain()
    results = []
    
    status = st.empty()
    progress = st.progress(0)
    
    for i, (area, comando_especifico) in enumerate(audit_protocol):
        status.markdown(f"**🕵️ Auditando Dimensão:** {area}...")
        
        # Busca Jurisprudência
        docs_lei = vectorstore.similarity_search(comando_especifico, k=5)
        
        final_query = f"""
        DOCUMENTO DO USUÁRIO (TEXTO COMPLETO):
        {doc_text}
        
        ORDEM DE AUDITORIA: 
        Dimensão: '{area}'.
        Foco: {comando_especifico}
        """
        
        try:
            response = chain.run(input_documents=docs_lei, question=final_query)
        except Exception as e:
            if "429" in str(e):
                response = "⚠️ O documento é muito extenso e atingiu o limite momentâneo da IA. Tente aguardar 1 minuto e tentar novamente."
            else:
                response = f"Erro técnico: {str(e)}"
        
        results.append((area, response))
        progress.progress((i + 1) / len(audit_protocol))
        
        # --- FREIO ABS ---
        # Pausa de 2 segundos para esfriar a API entre perguntas
        time.sleep(2)
    
    status.empty()
    return results

# --- 5. INTERFACE ---
def main():
    with st.sidebar:
        st.header("🔐 Acesso")
        if not st.session_state['logged']:
            key = st.text_input("Senha", type="password")
            if st.button("Entrar"):
                if check_login(key) > -1:
                    st.session_state['logged'] = True
                    st.session_state['user_key'] = key
                    st.rerun()
                else:
                    st.error("Negado")
        else:
            st.success(f"Auditor: {st.session_state.get('user_key')}")
            if st.button("Sair"):
                st.session_state['logged'] = False
                st.rerun()

    if st.session_state['logged']:
        st.title("🏛️ AUDITOR LICI TECHGOV - BY GUSTAVO (v6.1)")
        
        if 'vectorstore' not in st.session_state:
            with st.spinner("Carregando Cérebro Jurídico..."):
                vs, logs = load_knowledge_base()
                if vs: st.session_state['vectorstore'] = vs
                else: st.error("Erro na Base.")
        
        if st.session_state.get('vectorstore'):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.info("📂 Configuração da Auditoria")
                doc_type = st.radio("Tipo de Documento:", ["EDITAL", "ETP", "TR / PROJETO BÁSICO"])
                uploaded = st.file_uploader("Arquivo PDF", type="pdf")
                start = st.button("🔍 INICIAR VARREDURA TOTAL", type="primary")

            with col2:
                if uploaded and start:
                    
                    if doc_type == "EDITAL":
                        protocol = [
                            ("1. Legalidade, Objeto e Fundamentação", "Verifique legalidade do objeto, Lei 14.133 e critério de julgamento."),
                            ("2. Habilitação e Restrições (Pente-Fino)", "Analise RIGOROSAMENTE as cláusulas de habilitação. Busque restrições (sede local, vistoria obrigatória, índices abusivos, capital excessivo)."),
                            ("3. Orçamento, Reajuste e Financeiro", "Verifique orçamento, reajuste (obrigatório), aceitabilidade de preços e garantia."),
                            ("4. Ritos, Prazos e Recursos", "Verifique prazos de publicidade, impugnação, recurso e validade das propostas.")
                        ]
                    
                    elif doc_type == "ETP":
                        protocol = [
                            ("1. Necessidade e Planejamento", "Necessidade pública (Inc I) e PCA (Inc II)."),
                            ("2. Estudo de Mercado e Solução", "Levantamento de alternativas e estimativa de quantidades com memória."),
                            ("3. Parcelamento do Objeto", "Justificativa expressa para o parcelamento ou não (Súmula 247 TCU)."),
                            ("4. Viabilidade e Valor", "Estimativa de valor e conclusão de viabilidade.")
                        ]
                    
                    else: # TR / PB
                        protocol = [
                            ("1. Definição Técnica", "Descrição do objeto, quantitativos e referência ao ETP."),
                            ("2. Gestão e Fiscalização", "Modelo de gestão, indicação de fiscal/gestor e procedimentos."),
                            ("3. Pagamento e Recebimento", "Prazo de pagamento, critérios de medição e recebimento (provisório/definitivo)."),
                            ("4. Obrigações e Sanções", "Obrigações, garantia e sanções administrativas.")
                        ]

                    results = process_audit_full(st.session_state['vectorstore'], uploaded, protocol)
                    
                    st.subheader("📋 Relatório de Auditoria Completa")
                    for area, parecer in results:
                        if "OMISSÃO" in parecer or "ALERTA" in parecer or "ILEGAL" in parecer:
                             st.markdown(f"<div class='alert-box'><h3>{area}</h3>{parecer}</div>", unsafe_allow_html=True)
                        elif "CONFORME" in parecer or "ADEQUADO" in parecer:
                             st.markdown(f"<div class='success-box'><h3>{area}</h3>{parecer}</div>", unsafe_allow_html=True)
                        else:
                             st.markdown(f"<div class='neutral-box'><h3>{area}</h3>{parecer}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
