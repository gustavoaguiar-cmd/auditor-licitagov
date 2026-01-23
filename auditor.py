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
.warning-msg {
    color: #856404;
    background-color: #fff3cd;
    border-color: #ffeeba;
    padding: 10px;
    border-radius: 5px;
    font-size: 0.9em;
    margin-top: 5px;
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
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key, chunk_size=100)
    
    try:
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
        return vectorstore, files_log
    except Exception as e:
        return None, [f"ERRO CRÍTICO OPENAI: {str(e)}"]

# --- 3. CÉREBRO JURÍDICO (PROMPT INTELIGENTE) ---
def get_audit_chain():
    
    prompt_template = """
    Você é um Auditor Sênior Especialista em Licitações (Lei 14.133/21).
    
    Sua missão é evitar "Falsos Positivos".
    
    INSTRUÇÃO DE VARREDURA (Busca Holística):
    1. LEIA O TEXTO INTEIRO.
    2. Se você procura um requisito (ex: CNDT, Declaração PcD) e não encontrar na seção "Habilitação", BUSQUE NO RESTO DO DOCUMENTO (ex: Minuta de Contrato, Declarações Anexas).
    3. Se o item estiver presente em QUALQUER lugar do documento, considere ATENDIDO, mas faça uma ressalva se estiver no lugar errado.
    
    TEMA DA ANÁLISE: {question}
    
    CONTEXTO JURÍDICO:
    {context}
    
    PARECER DO AUDITOR:
    - Se achar irregularidade real: Comece com "🚨 ALERTA".
    - Se achar o item, mas em local estranho: Comece com "⚠️ RESSALVA" e explique (ex: "A CNDT é exigida no item 25 para pagamento, mas não consta na habilitação").
    - Se estiver tudo certo: Comece com "✅ CONFORME" e cite o item/página.
    - Omissão: Só use se tiver CERTEZA ABSOLUTA que não existe menção no arquivo inteiro.
    """

    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    # --- VOLTAMOS PARA O GPT-4o (O MAIS INTELIGENTE) ---
    # O handler de erro (backoff) vai gerenciar os limites.
    model = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# --- 4. FUNÇÃO ROBUSTA COM RETRY (SISTEMA ANTI-CRASH) ---
def run_with_retry(chain, docs_lei, final_query, max_retries=3):
    """Tenta rodar a IA. Se der erro de limite (429), espera e tenta de novo."""
    attempt = 0
    while attempt < max_retries:
        try:
            return chain.run(input_documents=docs_lei, question=final_query)
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "Rate limit" in error_msg:
                wait_time = 40 # Espera 40 segundos se bater no teto
                st.toast(f"⏳ Alto volume de dados. A IA está analisando profundamente... ({attempt+1}/{max_retries})")
                time.sleep(wait_time)
                attempt += 1
            else:
                return f"Erro técnico irrecuperável: {error_msg}"
    return "⚠️ Erro: O sistema da OpenAI está sobrecarregado no momento. Tente novamente em 2 minutos."

# --- 5. MOTOR DE AUDITORIA ---
def process_audit_full(vectorstore, uploaded_file, audit_protocol):
    reader = PdfReader(uploaded_file)
    doc_text = ""
    
    for i, page in enumerate(reader.pages):
        content = page.extract_text()
        if content:
            doc_text += f"\n--- PÁGINA {i+1} ---\n{content.replace(chr(0), '')}"
    
    if len(doc_text) < 50:
        return [("Erro", "Arquivo vazio.")]

    chain = get_audit_chain()
    results = []
    
    status = st.empty()
    progress = st.progress(0)
    
    # Aviso de Processamento Pesado
    st.info("ℹ️ Modo Análise Profunda ativado (GPT-4o). Isso pode levar alguns minutos para garantir precisão máxima.")
    
    for i, (area, comando_especifico) in enumerate(audit_protocol):
        status.markdown(f"**🕵️ Auditando Dimensão:** {area}...")
        
        docs_lei = vectorstore.similarity_search(comando_especifico, k=5)
        
        final_query = f"""
        DOCUMENTO DO USUÁRIO (TEXTO COMPLETO):
        {doc_text}
        
        ORDEM DE AUDITORIA: 
        Dimensão: '{area}'.
        Foco: {comando_especifico}
        """
        
        # Chama a função segura com retry
        response = run_with_retry(chain, docs_lei, final_query)
        
        results.append((area, response))
        progress.progress((i + 1) / len(audit_protocol))
        
    status.empty()
    return results

# --- 6. INTERFACE ---
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
        st.title("🏛️ AUDITOR LICI TECHGOV - BY GUSTAVO (v7.0)")
        
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
                            ("1. Legalidade, Objeto e Fundamentação", "Verifique legalidade do objeto e Lei 14.133."),
                            ("2. Habilitação e Restrições (Pente-Fino)", "Analise a Habilitação. IMPORTANTE: Antes de apontar omissão de CNDT ou Declarações, busque no documento inteiro (incluindo anexos e condições de execução)."),
                            ("3. Orçamento, Reajuste e Financeiro", "Verifique orçamento, reajuste e garantia."),
                            ("4. Ritos, Prazos e Recursos", "Verifique prazos e validade das propostas.")
                        ]
                    
                    elif doc_type == "ETP":
                        protocol = [
                            ("1. Necessidade e Planejamento", "Necessidade pública (Inc I) e PCA (Inc II)."),
                            ("2. Estudo de Mercado e Solução", "Levantamento de alternativas e estimativa de quantidades."),
                            ("3. Parcelamento do Objeto", "Justificativa expressa para o parcelamento ou não."),
                            ("4. Viabilidade e Valor", "Estimativa de valor e conclusão.")
                        ]
                    
                    else: # TR / PB
                        protocol = [
                            ("1. Definição Técnica", "Descrição do objeto e quantitativos."),
                            ("2. Gestão e Fiscalização", "Modelo de gestão e fiscalização."),
                            ("3. Pagamento e Recebimento", "Prazo de pagamento e critérios de medição."),
                            ("4. Obrigações e Sanções", "Obrigações e sanções.")
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
