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
    .alert-box { background-color: #ffe6e6; border-left: 6px solid #ff4b4b; padding: 15px; margin-bottom: 20px; border-radius: 5px; color: #333; }
    .success-box { background-color: #e6fffa; border-left: 6px solid #00cc99; padding: 15px; margin-bottom: 20px; border-radius: 5px; color: #333; }
    .neutral-box { background-color: #f0f2f6; border-left: 6px solid #555; padding: 15px; margin-bottom: 20px; border-radius: 5px; color: #333; }
    .warning-box { background-color: #fff3cd; border-left: 6px solid #ffecb5; padding: 15px; margin-bottom: 20px; border-radius: 5px; color: #664d03; }
    
    .landing-header { font-size: 3em; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 0.2em; text-transform: uppercase; letter-spacing: 2px; }
    .landing-sub { font-size: 1.4em; color: #555; text-align: center; margin-bottom: 3em; font-weight: 300; }
    .feature-card { background-color: #ffffff; padding: 30px; border-radius: 15px; box-shadow: 0 10px 20px rgba(0,0,0,0.08); text-align: center; height: 100%; border-top: 5px solid #1E3A8A; transition: transform 0.3s ease; }
    .feature-card:hover { transform: translateY(-5px); }
    [data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #dee2e6; }
</style>
""", unsafe_allow_html=True)

# --- SESSÃO ---
if 'logged' not in st.session_state: st.session_state['logged'] = False

# --- 1. LOGIN ---
def check_login(key):
    users = {"AMIGO_TESTE": 3, "PREFEITURA_X": 10, "GUSTAVO_ADMIN": 999}
    return users.get(key, -1)

# --- 2. CARREGAMENTO DA BASE (BATCHING) ---
@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    text = ""
    data_folder = "data"
    
    if not os.path.exists(data_folder):
        try: os.makedirs(data_folder); return None, ["⚠️ Pasta 'data' criada."]
        except: return None, ["❌ Erro de permissão."]

    files_log = []
    for root, dirs, files in os.walk(data_folder):
        for filename in files:
            if filename.lower().endswith('.pdf'):
                try:
                    pdf_reader = PdfReader(os.path.join(root, filename))
                    for page in pdf_reader.pages:
                        if page.extract_text(): text += f"\n[FONTE DE CONTEXTO: {filename}] {page.extract_text()}"
                    files_log.append(f"✅ Lido: {filename}")
                except: continue
    
    if not text: return None, ["⚠️ Base vazia ou sem OCR."]

    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        
        api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key: return None, ["❌ Sem API Key."]
        
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = None
        batch_size = 50 
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            if not batch: continue
            if vectorstore is None: vectorstore = FAISS.from_texts(batch, embeddings)
            else: vectorstore.add_texts(batch)
            time.sleep(0.2)
            
        return vectorstore, files_log
    except Exception as e:
        return None, [f"❌ Erro Técnico: {str(e)}"]

# --- 3. CÉREBRO JURÍDICO (PROMPT BLINDADO ANTI-ALUCINAÇÃO) ---
def create_chain(model_name="gpt-4o"):
    prompt_template = """
    Você é um Auditor Sênior Especialista em Licitações (Lei 14.133/21).
    
    ESTRUTURA DE ANÁLISE:
    1. DOC_ALVO: É o documento que o usuário fez upload. SUA ANÁLISE DEVE SER 100% BASEADA NELE.
    2. CONTEXTO_JURIDICO: São leis e acórdãos apenas para REFERÊNCIA COMPARATIVA.
    
    REGRA DE OURO (ANTI-ALUCINAÇÃO):
    - JAMAIS atribua fatos do CONTEXTO_JURIDICO ao DOC_ALVO. 
    - Se o CONTEXTO falar de "Medicamentos em José do Calçado" e o DOC_ALVO for sobre "Imigração", IGNORE o contexto e diga que o DOC_ALVO não trata de licitação.
    
    TEMA DA VARREDURA: {question}
    
    CONTEXTO_JURIDICO (Use APENAS como base legal, NÃO como fato do caso):
    {context}
    
    PARECER DO AUDITOR:
    - Se o DOC_ALVO não for pertinente ao tema (ex: documento de imigração, receita médica): Responda "⚠️ DOCUMENTO INVÁLIDO: O arquivo analisado não parece ser uma peça técnica de licitação."
    - Irregularidade no DOC_ALVO: "🚨 ALERTA".
    - Ressalva: "⚠️ RESSALVA".
    - Conforme: "✅ CONFORME".
    """
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = ChatOpenAI(model=model_name, temperature=0, openai_api_key=api_key)
    return load_qa_chain(model, chain_type="stuff", prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"]))

# --- 4. MOTOR HÍBRIDO ---
def smart_audit_run(vectorstore, final_query, docs_lei):
    try:
        chain = create_chain("gpt-4o")
        return chain.run(input_documents=docs_lei, question=final_query), "premium"
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "Request too large" in error_msg:
            try:
                chain_turbo = create_chain("gpt-4o-mini")
                return chain_turbo.run(input_documents=docs_lei, question=final_query), "turbo"
            except Exception as e2:
                return f"⚠️ Erro Crítico: Documento muito grande. (Erro: {str(e2)})", "error"
        elif "insufficient_quota" in error_msg:
            return "💸 FALHA: Saldo da OpenAI Esgotado.", "error"
        return f"⚠️ Erro Técnico: {error_msg}", "error"

# --- 5. PROCESSAMENTO ---
def process_audit_full(vectorstore, uploaded_file, audit_protocol):
    reader = PdfReader(uploaded_file)
    doc_text = ""
    for i, page in enumerate(reader.pages):
        if page.extract_text(): doc_text += f"\n--- PÁGINA {i+1} ---\n{page.extract_text()}"
    
    if len(doc_text) < 50: return [("Erro", "Arquivo vazio.")]

    results = []
    status = st.empty()
    progress = st.progress(0)
    
    st.info("🚀 Auditoria Iniciada. Cruzando dados com Jurisprudência (Sem contaminação).")
    
    for i, (area, comando) in enumerate(audit_protocol):
        status.markdown(f"**🕵️ Auditando:** {area}...")
        
        # Busca contexto mas o prompt agora sabe separar
        docs_lei = vectorstore.similarity_search(comando, k=5)
        
        # Deixamos CLARO o que é o documento do usuário
        final_query = f"DOC_ALVO (TEXTO DO USUÁRIO): {doc_text}\n\nTAREFA: Auditar o item '{area}' com foco em: {comando}"
        
        resp, motor = smart_audit_run(vectorstore, final_query, docs_lei)
        
        if motor == "turbo": resp += "\n\n*(Nota: Processado via Motor Turbo)*"
            
        results.append((area, resp))
        progress.progress((i + 1) / len(audit_protocol))
        time.sleep(1)
        
    status.empty()
    return results

# --- 6. INTERFACE ---
def main():
    with st.sidebar:
        st.markdown("### 🔐 Acesso Restrito")
        if not st.session_state['logged']:
            key = st.text_input("Chave de Acesso", type="password")
            if st.button("Entrar"):
                if check_login(key) > -1:
                    st.session_state['logged'] = True
                    st.session_state['user_key'] = key
                    st.rerun()
                else: st.error("Negado.")
        else:
            st.success(f"Logado: {st.session_state.get('user_key')}")
            if st.button("Sair"):
                st.session_state['logged'] = False
                st.rerun()

    if not st.session_state['logged']:
        st.markdown("<div class='landing-header'>🏛️ LICI TECHGOV</div>", unsafe_allow_html=True)
        st.markdown("<div class='landing-sub'>Inteligência Artificial de Alta Precisão para Gestão Pública</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown("""<div class='feature-card'><h4>🔍 Auditoria 360º</h4><p>Lei 14.133/21 + Jurisprudência TCU/TCE.</p></div>""", unsafe_allow_html=True)
        with c2: st.markdown("""<div class='feature-card'><h4>⚡ IA Premium</h4><p>Motor Híbrido Anti-Falha (GPT-4o).</p></div>""", unsafe_allow_html=True)
        with c3: st.markdown("""<div class='feature-card'><h4>🛡️ Blindagem</h4><p>Reduza impugnações com análise preditiva.</p></div>""", unsafe_allow_html=True)

    else:
        st.title("🏛️ AUDITOR LICI TECHGOV (v9.4)")
        
        if 'vectorstore' not in st.session_state:
            with st.spinner("Carregando Base Jurídica..."):
                vs, logs = load_knowledge_base()
                if vs: st.session_state['vectorstore'] = vs
                else: 
                    st.error("Base não carregada.")
                    with st.expander("Logs"):
                        for log in logs: st.write(log)
        
        if st.session_state.get('vectorstore'):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.info("📂 Configuração")
                # MENU ATUALIZADO COM TR E PROJETO BÁSICO SEPARADOS
                doc_type = st.radio("Tipo de Documento:", 
                                  ["EDITAL", "ETP", "TR (Bens/Serviços)", "PROJETO BÁSICO (Obras/Engenharia)"])
                
                uploaded = st.file_uploader("Arquivo PDF", type="pdf")
                start = st.button("🔍 EXECUTAR AUDITORIA", type="primary")

            with col2:
                if uploaded and start:
                    
                    # PROTOCOLOS ATUALIZADOS
                    if doc_type == "EDITAL":
                        prot = [("1. Legalidade", "Legalidade objeto e Lei 14.133."), ("2. Habilitação", "Varredura CNDT/PcD/Balanço."), ("3. Financeiro", "Orçamento/Garantia."), ("4. Ritos", "Prazos.")]
                    
                    elif doc_type == "ETP":
                        prot = [("1. Necessidade", "PCA e Interesse Público."), ("2. Solução", "Estudo de Mercado e Alternativas."), ("3. Parcelamento", "Justificativa Súmula 247."), ("4. Viabilidade", "Estimativa Valor.")]
                    
                    elif "TR" in doc_type: # TR (Bens/Serviços Comuns)
                        prot = [
                            ("1. Definição do Objeto", "Especificação técnica, vedação a marca (ou justificativa) e quantitativos."),
                            ("2. Gestão do Contrato", "Fiscalização, recebimento provisório/definitivo e prazos."),
                            ("3. Pagamento e Sanções", "Critérios de medição, pagamento e rol de sanções."),
                            ("4. Exigências Técnicas", "Qualificação técnica compatível e amostras (se houver).")
                        ]
                        
                    elif "PROJETO BÁSICO" in doc_type: # Obras/Engenharia
                        prot = [
                            ("1. Engenharia e Custos", "Cronograma físico-financeiro, BDI detalhado e Planilha Orçamentária (SINAPI/SICRO)."),
                            ("2. Licenciamento e Matriz de Risco", "Licenciamento ambiental, desapropriações e Matriz de Riscos (Obrigatório)."),
                            ("3. Qualificação Técnica", "Atestados, CAT e visita técnica (justificada)."),
                            ("4. Execução e Medição", "Critérios de medição, reajustamento e subcontratação.")
                        ]

                    res = process_audit_full(st.session_state['vectorstore'], uploaded, prot)
                    
                    st.subheader("📋 Relatório")
                    for a, t in res:
                        if "INVÁLIDO" in t: st.markdown(f"<div class='neutral-box'><h3>{a}</h3>{t}</div>", unsafe_allow_html=True)
                        elif "ALERTA" in t: st.markdown(f"<div class='alert-box'><h3>{a}</h3>{t}</div>", unsafe_allow_html=True)
                        elif "RESSALVA" in t: st.markdown(f"<div class='warning-box'><h3>{a}</h3>{t}</div>", unsafe_allow_html=True)
                        elif "CONFORME" in t: st.markdown(f"<div class='success-box'><h3>{a}</h3>{t}</div>", unsafe_allow_html=True)
                        else: st.markdown(f"<div class='neutral-box'><h3>{a}</h3>{t}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
