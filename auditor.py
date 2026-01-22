import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="AUDITOR LICI TECHGOV", page_icon="⚖️", layout="wide")

# --- CSS PARA RELATÓRIOS PROFISSIONAIS ---
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

# --- INICIALIZAR VARIÁVEIS DE SESSÃO ---
if 'logged' not in st.session_state: st.session_state['logged'] = False

# --- 1. FUNÇÃO DE LOGIN ---
def check_login(key):
    users = {
        "AMIGO_TESTE": 3,
        "PREFEITURA_X": 10,
        "GUSTAVO_ADMIN": 999
    }
    return users.get(key, -1)

# --- 2. CARREGAMENTO DA BASE LEGAL (LEI + JURISPRUDÊNCIA) ---
@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    text = ""
    data_folder = "data"
    
    if not os.path.exists(data_folder):
        return None, ["ERRO CRÍTICO: Pasta 'data' não encontrada no sistema."]

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
                            # Limpeza de caracteres nulos
                            clean_page = page_text.replace('\x00', '')
                            text += f"\n[FONTE JURÍDICA: {filename}] {clean_page}"
                    files_log.append(f"✅ Base Carregada: {filename}")
                except Exception:
                    files_log.append(f"❌ Erro ao ler base: {filename}")
                    continue
    
    if text == "": return None, files_log

    # Splitter inteligente otimizado para leis
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

# --- 3. CÉREBRO JURÍDICO (PROMPTS DE VARREDURA TOTAL) ---
def get_audit_chain():
    
    # PROMPT GENÉRICO E PODEROSO - O "CÃO DE GUARDA"
    prompt_template = """
    Você é um Auditor Sênior Especialista em Licitações Públicas (Lei 14.133/21).
    Sua tarefa é auditar o documento fornecido minuciosamente, do início ao fim.
    
    INSTRUÇÃO DE VARREDURA:
    1. LEIA O TEXTO INTEIRO. Não pare na metade. Informações vitais (garantia, pagamento) podem estar no final.
    2. Cruze o texto do documento com o CONTEXTO JURÍDICO fornecido (Leis, Súmulas TCU, Acórdãos).
    3. Identifique TODAS as irregularidades, restrições indevidas, omissões obrigatórias ou cláusulas vagas.
    4. Se o texto estiver correto e completo, confirme citando o item/página onde encontrou a informação.
    
    TEMA DA ANÁLISE (Onde focar sua lupa agora): {question}
    
    CONTEXTO JURÍDICO (Sua Base de Conhecimento):
    {context}
    
    PARECER DO AUDITOR:
    - Seja rigoroso. Aponte o Artigo da Lei ou Súmula violada.
    - Se houver exigência restritiva (ex: limitação geográfica, taxas ilegais, excesso de atestados), denuncie.
    - Se faltar algo essencial (ex: BDI em obras, Reajuste, Fiscalização), aponte como OMISSÃO GRAVE.
    - Se estiver tudo certo, diga "CONFORME" e explique porquê.
    """

    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    # GPT-4o COM MAXIMA INTELIGENCIA (Sem limites de token)
    model = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# --- 4. MOTOR DE AUDITORIA (LEITURA INTEGRAL) ---
def process_audit_full(vectorstore, uploaded_file, audit_protocol):
    reader = PdfReader(uploaded_file)
    doc_text = ""
    
    # 1. LEITURA COMPLETA DO ARQUIVO (Página a Página)
    for i, page in enumerate(reader.pages):
        content = page.extract_text()
        if content:
            doc_text += f"\n--- PÁGINA {i+1} ---\n{content.replace(chr(0), '')}"
    
    if len(doc_text) < 50:
        return [("Erro", "Arquivo vazio ou ilegível.")]

    chain = get_audit_chain()
    results = []
    
    status = st.empty()
    progress = st.progress(0)
    
    full_report_text = ""
    
    # 2. LOOP PELO PROTOCOLO (VARREDURA POR DIMENSÕES)
    for i, (area, comando_especifico) in enumerate(audit_protocol):
        status.markdown(f"**🕵️ Auditando Dimensão:** {area}...")
        
        # Busca Jurisprudência Relevante para esta dimensão na base
        docs_lei = vectorstore.similarity_search(comando_especifico, k=5)
        
        # Monta o Prompt com o DOCUMENTO INTEIRO
        final_query = f"""
        DOCUMENTO DO USUÁRIO (TEXTO COMPLETO PARA ANÁLISE):
        {doc_text}
        
        --------------------------------------------------
        
        ORDEM DE AUDITORIA: 
        Dimensão: '{area}'.
        O que buscar: {comando_especifico}
        
        Verifique se há conformidade total ou se há vícios.
        """
        
        try:
            response = chain.run(input_documents=docs_lei, question=final_query)
        except Exception as e:
            response = f"Erro técnico: {str(e)}"
        
        results.append((area, response))
        full_report_text += f"\n\nDIMENSÃO {area}:\n{response}"
        progress.progress((i + 1) / len(audit_protocol))
    
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
        st.title("🏛️ AUDITOR LICI TECHGOV - BY GUSTAVO (v6.0)")
        
        if 'vectorstore' not in st.session_state:
            with st.spinner("Carregando Cérebro Jurídico (Leis + TCU)..."):
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
                    
                    # --- O PROTOCOLO DE VARREDURA ---
                    # Essas são as "lentes" que o robô vai usar para ler o texto inteiro.
                    # Elas cobrem TODAS as áreas da lei, garantindo que nada passe batido.
                    
                    if doc_type == "EDITAL":
                        protocol = [
                            ("1. Legalidade, Objeto e Fundamentação", "Verifique a legalidade do objeto, se há definição clara, se cita a Lei 14.133 corretamente e se o critério de julgamento está adequado."),
                            ("2. Habilitação e Restrições (Pente-Fino)", "Analise RIGOROSAMENTE as cláusulas de habilitação. Busque por exigências que restrinjam a competição (sede local, vistoria obrigatória, índices abusivos, capital excessivo)."),
                            ("3. Orçamento, Reajuste e Financeiro", "Verifique as regras de orçamento, cláusula de reajuste (obrigatória), critérios de aceitabilidade de preços e garantia."),
                            ("4. Ritos, Prazos e Recursos", "Verifique se os prazos de publicidade, impugnação, recurso e validade das propostas respeitam a Lei 14.133.")
                        ]
                    
                    elif doc_type == "ETP":
                        protocol = [
                            ("1. Necessidade e Planejamento (Inc. I e II)", "Verifique se a necessidade pública está justificada e se há previsão no PCA."),
                            ("2. Estudo de Mercado e Solução (Inc. V, VI, VII)", "Analise se houve levantamento de alternativas, estimativa de quantidades e definição da solução."),
                            ("3. Parcelamento do Objeto (Inc. VIII)", "Verifique se há justificativa expressa para o parcelamento ou não (Súmula 247 TCU). Item CRÍTICO."),
                            ("4. Viabilidade e Conclusão (Inc. XIII)", "Verifique a estimativa de valor e o posicionamento conclusivo sobre a viabilidade.")
                        ]
                    
                    else: # TR / PB
                        protocol = [
                            ("1. Definição Técnica e Objeto", "Analise a descrição do objeto, quantitativos, se bate com o ETP."),
                            ("2. Gestão e Fiscalização (Fiscal e Gestor)", "Verifique se há modelo de gestão, indicação de fiscal/gestor e procedimentos de fiscalização."),
                            ("3. Pagamento, Medição e Recebimento", "Analise CRITERIOSAMENTE: Prazo de pagamento, critérios de medição e recebimento (provisório/definitivo)."),
                            ("4. Obrigações, Garantia e Sanções", "Verifique as obrigações da contratada, prazo de garantia (CDC/Lei) e sanções administrativas.")
                        ]

                    # RODA A AUDITORIA TOTAL
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
