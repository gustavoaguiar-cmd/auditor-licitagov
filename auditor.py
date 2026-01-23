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
st.set_page_config(page_title="LICI TECHGOV", page_icon="🏛️", layout="wide")

# --- CSS VISUAL PROFISSIONAL ---
st.markdown("""
<style>
    /* Estilos Gerais */
    .alert-box { background-color: #ffe6e6; border-left: 6px solid #ff4b4b; padding: 15px; margin-bottom: 20px; border-radius: 5px; color: #333; }
    .success-box { background-color: #e6fffa; border-left: 6px solid #00cc99; padding: 15px; margin-bottom: 20px; border-radius: 5px; color: #333; }
    .neutral-box { background-color: #f0f2f6; border-left: 6px solid #555; padding: 15px; margin-bottom: 20px; border-radius: 5px; color: #333; }
    .status-msg { font-size: 0.9em; color: #155724; background-color: #d4edda; border-color: #c3e6cb; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
    
    /* Landing Page */
    .landing-header { font-size: 3em; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 0.5em; }
    .landing-sub { font-size: 1.5em; color: #555; text-align: center; margin-bottom: 2em; }
    .feature-card { background-color: #fff; padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; height: 100%; }
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

# --- 3. CONSTRUTOR DE CÉREBROS (PROMPT FACTORY) ---
def create_chain():
    """Cria a inteligência do auditor SEMPRE com o melhor modelo (GPT-4o)"""
    prompt_template = """
    Você é um Auditor Sênior Especialista em Licitações (Lei 14.133/21).
    
    INSTRUÇÃO DE VARREDURA (Busca Holística):
    1. LEIA O TEXTO INTEIRO.
    2. Se procurar um requisito (ex: CNDT, PcD) e não achar na "Habilitação", BUSQUE NO RESTO DO DOCUMENTO (ex: Minuta, Pagamento).
    3. Se o item existir em QUALQUER lugar, considere ATENDIDO com ressalva.
    
    TEMA DA ANÁLISE: {question}
    CONTEXTO JURÍDICO: {context}
    
    PARECER:
    - Irregularidade real: "🚨 ALERTA".
    - Item deslocado/estranho: "⚠️ RESSALVA" (Explique onde achou).
    - Conforme: "✅ CONFORME" (Cite o item).
    """
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    # FORÇANDO O GPT-4o (O MAIS INTELIGENTE)
    model = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# --- 4. MOTOR ROBUSTO (FILA DE ESPERA) ---
def robust_audit_run(vectorstore, final_query, docs_lei):
    """
    Tenta rodar com GPT-4o. 
    Se der erro de limite (429), ele ESPERA e TENTA DE NOVO.
    Não rebaixa para o Mini.
    """
    chain = create_chain()
    max_retries = 5 # Tenta até 5 vezes
    base_wait = 20 # Começa esperando 20 segundos
    
    for attempt in range(max_retries):
        try:
            return chain.run(input_documents=docs_lei, question=final_query)
        
        except Exception as e:
            error_msg = str(e)
            # Se for erro de taxa (Rate Limit)
            if "429" in error_msg or "Rate limit" in error_msg:
                wait_time = base_wait * (attempt + 1) # Aumenta o tempo a cada erro (20s, 40s, 60s...)
                st.toast(f"⏳ Servidor cheio. Aguardando {wait_time}s para garantir análise Premium... (Tentativa {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                # Se for outro erro, avisa
                return f"Erro técnico na análise: {error_msg}"
    
    return "⚠️ O sistema da OpenAI está muito congestionado. Tente novamente em 5 minutos ou verifique seus créditos."

# --- 5. PROCESSAMENTO PRINCIPAL ---
def process_audit_full(vectorstore, uploaded_file, audit_protocol):
    reader = PdfReader(uploaded_file)
    doc_text = ""
    for i, page in enumerate(reader.pages):
        content = page.extract_text()
        if content: doc_text += f"\n--- PÁGINA {i+1} ---\n{content.replace(chr(0), '')}"
    
    if len(doc_text) < 50: return [("Erro", "Arquivo vazio.")]

    results = []
    status = st.empty()
    progress = st.progress(0)
    
    st.info("🚀 Auditoria Premium Iniciada: Usando GPT-4o (Alta Precisão). Pode haver pausas para processamento.")
    
    for i, (area, comando_especifico) in enumerate(audit_protocol):
        status.markdown(f"**🕵️ Auditando Dimensão:** {area}...")
        
        docs_lei = vectorstore.similarity_search(comando_especifico, k=5)
        
        final_query = f"""
        DOCUMENTO DO USUÁRIO (TEXTO COMPLETO): {doc_text}
        ORDEM DE AUDITORIA: Dimensão '{area}'. Foco: {comando_especifico}
        """
        
        # Chama o Motor Robusto
        resposta = robust_audit_run(vectorstore, final_query, docs_lei)
            
        results.append((area, resposta))
        progress.progress((i + 1) / len(audit_protocol))
        
    status.empty()
    return results

# --- 6. INTERFACE ---
def main():
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("### 🔐 Acesso Restrito")
        if not st.session_state['logged']:
            key = st.text_input("Chave de Acesso", type="password")
            if st.button("Acessar Painel"):
                if check_login(key) > -1:
                    st.session_state['logged'] = True
                    st.session_state['user_key'] = key
                    st.rerun()
                else:
                    st.error("Chave não reconhecida.")
        else:
            st.success(f"Usuário: {st.session_state.get('user_key')}")
            if st.button("Sair / Logout"):
                st.session_state['logged'] = False
                st.rerun()
            st.markdown("---")
            st.caption("Licenciado para: AguiarGov")

    # --- MAIN CONTENT ---
    if not st.session_state['logged']:
        # LANDING PAGE
        st.markdown("<div class='landing-header'>🏛️ LICI TECHGOV</div>", unsafe_allow_html=True)
        st.markdown("<div class='landing-sub'>A Primeira IA Auditora de Contratações Públicas do Brasil</div>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("<div class='feature-card'><h4>🔍 Auditoria Jurídica</h4><p>Varredura completa baseada na Lei 14.133/21.</p></div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='feature-card'><h4>⚡ Inteligência Premium</h4><p>Análise de profundidade com GPT-4o.</p></div>", unsafe_allow_html=True)
        with c3:
            st.markdown("<div class='feature-card'><h4>📊 Relatórios Premium</h4><p>Pareceres prontos para a Procuradoria.</p></div>", unsafe_allow_html=True)

    else:
        # DASHBOARD
        st.title("🏛️ AUDITOR LICI TECHGOV (v8.1 Premium)")
        st.markdown(f"Olá, **{st.session_state.get('user_key')}**. O sistema está configurado para precisão máxima.")
        st.markdown("---")
        
        if 'vectorstore' not in st.session_state:
            with st.spinner("Carregando Base Legal e Jurisprudência..."):
                vs, logs = load_knowledge_base()
                if vs: st.session_state['vectorstore'] = vs
                else: st.error("Falha ao iniciar base de dados.")
        
        if st.session_state.get('vectorstore'):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.info("📂 Parâmetros da Análise")
                doc_type = st.radio("Selecione o Tipo:", ["EDITAL", "ETP", "TR / PROJETO BÁSICO"])
                uploaded = st.file_uploader("Upload do Arquivo (PDF)", type="pdf")
                start = st.button("🔍 EXECUTAR AUDITORIA", type="primary", use_container_width=True)

            with col2:
                if uploaded and start:
                    
                    # Protocolos Rigorosos
                    if doc_type == "EDITAL":
                        protocol = [
                            ("1. Legalidade e Fundamentação", "Verifique legalidade do objeto e Lei 14.133."),
                            ("2. Habilitação (Varredura Total)", "Analise Habilitação. IMPORTANTE: Antes de apontar omissão (CNDT, PcD), busque no ARQUIVO INTEIRO."),
                            ("3. Financeiro e Garantias", "Verifique orçamento, reajuste e garantias."),
                            ("4. Ritos e Prazos", "Verifique prazos e validade das propostas.")
                        ]
                    elif doc_type == "ETP":
                        protocol = [("1. Necessidade e PCA", "Necessidade pública e PCA."), ("2. Solução", "Alternativas e estimativa."), ("3. Parcelamento", "Justificativa (Súmula 247 TCU)."), ("4. Viabilidade", "Conclusão.")]
                    else:
                        protocol = [("1. Técnica", "Objeto e quantitativos."), ("2. Gestão", "Fiscalização."), ("3. Pagamento", "Medição e pagamento."), ("4. Sanções", "Obrigações e sanções.")]

                    results = process_audit_full(st.session_state['vectorstore'], uploaded, protocol)
                    
                    st.subheader("📋 Relatório Final da Auditoria")
                    for area, parecer in results:
                        if "OMISSÃO" in parecer or "ALERTA" in parecer or "ILEGAL" in parecer:
                             st.markdown(f"<div class='alert-box'><h3>{area}</h3>{parecer}</div>", unsafe_allow_html=True)
                        elif "CONFORME" in parecer or "ADEQUADO" in parecer:
                             st.markdown(f"<div class='success-box'><h3>{area}</h3>{parecer}</div>", unsafe_allow_html=True)
                        else:
                             st.markdown(f"<div class='neutral-box'><h3>{area}</h3>{parecer}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
