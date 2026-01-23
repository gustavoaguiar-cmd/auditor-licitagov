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
    
    /* Landing Page - Design Premium */
    .landing-header { font-size: 3em; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 0.2em; text-transform: uppercase; letter-spacing: 2px; }
    .landing-sub { font-size: 1.4em; color: #555; text-align: center; margin-bottom: 3em; font-weight: 300; }
    
    .feature-card { 
        background-color: #ffffff; 
        padding: 30px; 
        border-radius: 15px; 
        box-shadow: 0 10px 20px rgba(0,0,0,0.08); 
        text-align: center; 
        height: 100%; 
        border-top: 5px solid #1E3A8A;
        transition: transform 0.3s ease;
    }
    .feature-card:hover { transform: translateY(-5px); }
    .feature-card h4 { color: #1E3A8A; font-weight: bold; font-size: 1.2em; margin-bottom: 15px; }
    .feature-card p { color: #666; font-size: 1em; line-height: 1.6; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #dee2e6; }
</style>
""", unsafe_allow_html=True)

# --- SESSÃO ---
if 'logged' not in st.session_state: st.session_state['logged'] = False

# --- 1. LOGIN ---
def check_login(key):
    users = {"AMIGO_TESTE": 3, "PREFEITURA_X": 10, "GUSTAVO_ADMIN": 999}
    return users.get(key, -1)

# --- 2. CARREGAMENTO DA BASE (COM PROTEÇÃO DE COTA) ---
@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    text = ""
    data_folder = "data"
    
    # Cria pasta se não existir
    if not os.path.exists(data_folder):
        try:
            os.makedirs(data_folder)
            return None, ["⚠️ Pasta 'data' criada. Adicione os PDFs jurídicos nela."]
        except:
            return None, ["❌ Erro de permissão ao criar pasta."]

    files_log = []
    pdf_count = 0
    for root, dirs, files in os.walk(data_folder):
        for filename in files:
            if filename.lower().endswith('.pdf'):
                pdf_count += 1
                try:
                    pdf_reader = PdfReader(os.path.join(root, filename))
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            clean_page = page_text.replace('\x00', '')
                            text += f"\n[FONTE JURÍDICA: {filename}] {clean_page}"
                    files_log.append(f"✅ Indexado: {filename}")
                except Exception:
                    files_log.append(f"❌ Falha: {filename}")
                    continue
    
    if pdf_count == 0: return None, ["⚠️ Nenhum PDF na pasta 'data'."]
    if text == "": return None, ["⚠️ PDFs vazios ou sem OCR."]

    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200, separators=["\nArt.", "\n\n", ". ", " ", ""])
        chunks = [c for c in text_splitter.split_text(text) if c and len(c.strip()) > 20]
        
        api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key: return None, ["❌ Chave API não encontrada."]
        
        embeddings = OpenAIEmbeddings(openai_api_key=api_key, chunk_size=1000)
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
        return vectorstore, files_log
        
    except Exception as e:
        error_msg = str(e)
        if "insufficient_quota" in error_msg:
            return None, ["💸 ERRO DE SALDO: Sua conta OpenAI está sem créditos. Adicione fundos em platform.openai.com."]
        return None, [f"❌ Erro Técnico: {error_msg}"]

# --- 3. CÉREBRO ---
def create_chain():
    prompt_template = """
    Você é um Auditor Sênior Especialista em Licitações (Lei 14.133/21).
    INSTRUÇÃO DE VARREDURA:
    1. LEIA O TEXTO INTEIRO.
    2. Se não achar requisito na Habilitação, busque no resto do documento (Minuta, Anexos, TR).
    3. Cruze com a Jurisprudência fornecida.
    
    TEMA: {question}
    CONTEXTO JURÍDICO: {context}
    
    PARECER:
    - Irregularidade Grave: "🚨 ALERTA".
    - Ressalva (Item deslocado): "⚠️ RESSALVA" (Explique onde encontrou).
    - Conforme: "✅ CONFORME" (Cite o item).
    """
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)
    return load_qa_chain(model, chain_type="stuff", prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"]))

# --- 4. MOTOR ROBUSTO ---
def robust_audit_run(vectorstore, final_query, docs_lei):
    chain = create_chain()
    for attempt in range(3):
        try:
            return chain.run(input_documents=docs_lei, question=final_query)
        except Exception as e:
            if "insufficient_quota" in str(e): return "💸 FALHA CRÍTICA: Saldo da OpenAI esgotado."
            if "429" in str(e):
                time.sleep(10) # Espera 10s e tenta de novo
                continue
            return f"Erro: {str(e)}"
    return "⚠️ Sistema ocupado. Tente novamente."

# --- 5. PROCESSAMENTO ---
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
    st.info("🚀 Auditoria em andamento (GPT-4o).")
    
    for i, (area, comando) in enumerate(audit_protocol):
        status.markdown(f"**🕵️ Auditando:** {area}...")
        docs_lei = vectorstore.similarity_search(comando, k=5)
        final_query = f"DOCUMENTO: {doc_text}\n\nAUDITORIA: '{area}'. Foco: {comando}"
        
        resp = robust_audit_run(vectorstore, final_query, docs_lei)
        results.append((area, resp))
        progress.progress((i + 1) / len(audit_protocol))
        
    status.empty()
    return results

# --- 6. INTERFACE ---
def main():
    with st.sidebar:
        st.markdown("### 🔐 Acesso Restrito")
        if not st.session_state['logged']:
            key = st.text_input("Chave de Acesso", type="password")
            if st.button("Entrar no Sistema"):
                if check_login(key) > -1:
                    st.session_state['logged'] = True
                    st.session_state['user_key'] = key
                    st.rerun()
                else: st.error("Credencial Inválida.")
        else:
            st.success(f"Logado: {st.session_state.get('user_key')}")
            if st.button("Sair"):
                st.session_state['logged'] = False
                st.rerun()
            st.markdown("---")
            st.caption("Licença Corporativa: AguiarGov")

    if not st.session_state['logged']:
        # LANDING PAGE COM REDAÇÃO DE VENDA (V9.0)
        st.markdown("<div class='landing-header'>🏛️ LICI TECHGOV</div>", unsafe_allow_html=True)
        st.markdown("<div class='landing-sub'>Inteligência Artificial de Alta Precisão para Gestão Pública</div>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class='feature-card'>
            <h4>🔍 Auditoria Jurídica 360º</h4>
            <p>Varredura completa baseada na <strong>Lei 14.133/21</strong> e cruzamento em tempo real com a <strong>Jurisprudência do TCU/TCE</strong>.</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class='feature-card'>
            <h4>⚡ Inteligência Artificial Premium</h4>
            <p>Motor GPT-4o calibrado para identificar riscos ocultos, omissões de garantias e cláusulas restritivas.</p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class='feature-card'>
            <h4>🛡️ Segurança Jurídica e Blindagem</h4>
            <p>Garanta editais robustos e reduza impugnações com análises preditivas antes da publicação.</p>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.title("🏛️ AUDITOR LICI TECHGOV (v9.0)")
        
        if 'vectorstore' not in st.session_state:
            with st.spinner("Inicializando Base de Conhecimento Jurídico..."):
                vs, logs = load_knowledge_base()
                if vs: st.session_state['vectorstore'] = vs
                else: 
                    st.error("Erro na Base de Dados.")
                    with st.expander("Ver Detalhes"):
                        for log in logs: st.write(log)
        
        if st.session_state.get('vectorstore'):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.info("📂 Parâmetros da Análise")
                doc_type = st.radio("Tipo de Documento:", ["EDITAL", "ETP", "TR / PROJETO BÁSICO"])
                uploaded = st.file_uploader("Upload do Arquivo (PDF)", type="pdf")
                start = st.button("🔍 EXECUTAR VARREDURA", type="primary")

            with col2:
                if uploaded and start:
                    if doc_type == "EDITAL":
                        prot = [
                            ("1. Legalidade e Fundamentação", "Verifique legalidade do objeto, Lei 14.133/21 e Jurisprudência."), 
                            ("2. Habilitação (Varredura Total)", "Analise Habilitação. Busque CNDT, PcD e Balanço no DOCUMENTO INTEIRO antes de apontar omissão."), 
                            ("3. Financeiro e Garantias", "Verifique orçamento, reajuste e garantias."), 
                            ("4. Ritos e Prazos", "Verifique prazos e validade das propostas.")
                        ]
                    elif doc_type == "ETP":
                        prot = [("1. Necessidade e PCA", "Necessidade pública e PCA."), ("2. Solução", "Alternativas e estimativa."), ("3. Parcelamento", "Justificativa (Súmula 247)."), ("4. Viabilidade", "Valor e Conclusão.")]
                    else:
                        prot = [("1. Técnica", "Objeto e quantitativos."), ("2. Gestão", "Fiscalização."), ("3. Pagamento", "Medição e pagamento."), ("4. Sanções", "Obrigações e sanções.")]

                    res = process_audit_full(st.session_state['vectorstore'], uploaded, prot)
                    
                    st.subheader("📋 Relatório de Auditoria")
                    for a, t in res:
                        if "ALERTA" in t or "FALHA" in t: st.markdown(f"<div class='alert-box'><h3>{a}</h3>{t}</div>", unsafe_allow_html=True)
                        elif "CONFORME" in t: st.markdown(f"<div class='success-box'><h3>{a}</h3>{t}</div>", unsafe_allow_html=True)
                        else: st.markdown(f"<div class='neutral-box'><h3>{a}</h3>{t}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
