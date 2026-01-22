import streamlit as st
import os
from PyPDF2 import PdfReader
# TROCAMOS O SPLITTER PARA UM MAIS ROBUSTO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Configuração da Página
st.set_page_config(page_title="AguiarGov - Auditor Fiscal", page_icon="⚖️", layout="wide")

# --- CSS PARA ALERTAS VERMELHOS ---
st.markdown("""
<style>
.alert-box {
    background-color: #ffdddd;
    border-left: 6px solid #f44336;
    padding: 10px;
    margin-bottom: 15px;
    color: #333;
}
</style>
""", unsafe_allow_html=True)

# --- INICIALIZAR MEMÓRIA ---
if 'result_edital' not in st.session_state: st.session_state['result_edital'] = None
if 'result_etp' not in st.session_state: st.session_state['result_etp'] = None
if 'result_tr' not in st.session_state: st.session_state['result_tr'] = None
if 'logged' not in st.session_state: st.session_state['logged'] = False
if 'user_credits' not in st.session_state: st.session_state['user_credits'] = 0

# --- 1. FUNÇÃO DE LOGIN ---
def check_login(key):
    users = {
        "AMIGO_TESTE": 3,
        "PREFEITURA_X": 10,
        "GUSTAVO_ADMIN": 99
    }
    return users.get(key, -1)

# --- 2. CARREGAMENTO DUPLO (LEI + JURISPRUDÊNCIA) ---
@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    text = ""
    data_folder = "data"
    
    if not os.path.exists(data_folder):
        return None, ["ERRO: Pasta 'data' não encontrada."]

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
                            # Adiciona metadados manuais no texto para a IA saber a fonte
                            text += f"\n[FONTE: {filename}] {page_text}"
                    files_log.append(f"✅ Lido: {filename}")
                except Exception:
                    files_log.append(f"❌ Erro ao ler: {filename}")
                    continue
    
    if text == "": return None, files_log

    # CORREÇÃO DO ERRO (Bad Request): USANDO RECURSIVE E FILTRANDO VAZIOS
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks_raw = text_splitter.split_text(text)
    
    # FILTRO CRÍTICO: REMOVE CHUNKS VAZIOS QUE TRAVAM A API
    chunks = [c for c in chunks_raw if c.strip()]
    
    if not chunks:
        return None, ["ERRO: Texto vazio após processamento."]

    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key: return None, ["ERRO: Chave API ausente."]
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore, files_log

# --- 3. CÉREBRO JURÍDICO (PROMPTS RIGOROSOS) ---
def get_specialized_chain(doc_type):
    
    # PROMPT DO EDITAL (COM AS SUAS REGRAS)
    if doc_type == "EDITAL":
        prompt_template = """
        Você é um Auditor de Controle Externo do Tribunal de Contas (TCU/TCE).
        Sua missão é blindar o EDITAL de licitação.
        
        REGRAS DE OURO (CHECKLIST):
        1. Aspectos Legais: Objeto claro, critério de julgamento, lei regente (14.133), minuta de contrato anexa.
        2. Habilitação (Art. 62-70): Exigências devem ser PROPORCIONAIS. 
           - ALERTA: Sede no município, vistoria obrigatória sem justificativa ou capital social > 10% são ILEGAIS.
           - Certidões: Apenas as previstas em lei.
        3. Orçamento: Se não estiver no edital, verifique se há menção ao Anexo/TR. Não diga que é ilegal se estiver referenciado.
        4. JURISPRUDÊNCIA: Cite Acórdãos do TCU ou Prejulgados do TCE/ES se houver violação.
        
        PERGUNTA: {question}
        
        CONTEXTO LEGAL E DO DOCUMENTO:
        {context}
        
        PARECER DO AUDITOR:
        - Responda de forma direta.
        - Se identificar cláusula restritiva, inicie com "🚨 ALERTA VERMELHO:".
        - Cite o artigo da Lei 14.133 violado ou atendido.
        - Se faltar jurisprudência no contexto, use seu conhecimento de base sobre Súmulas do TCU.
        """

    # PROMPT DO ETP (ART. 18 NA VEIA)
    elif doc_type == "ETP":
        prompt_template = """
        Você é um Auditor de Planejamento.
        Analise o ETP estritamente conforme o Art. 18, §1º da Lei 14.133/21.
        
        ITENS OBRIGATÓRIOS:
        I - Necessidade (Interesse Público)
        II - Previsão no Plano Anual (PCA)
        VI - Estimativa de Valor (com memória)
        VIII - Justificativa de Parcelamento (Súmula 247 TCU)
        XIII - Posicionamento Conclusivo
        
        PERGUNTA: {question}
        
        CONTEXTO: {context}
        
        PARECER:
        Verifique se o texto atende ao inciso. Se for vago, critique. Cite a jurisprudência se aplicável.
        """

    # PROMPT DO TR (ART. 6 XXIII)
    elif doc_type == "TR":
        prompt_template = """
        Você é um Auditor de Licitações.
        Analise o TR conforme Art. 6º, XXIII da Lei 14.133/21.
        
        VERIFIQUE:
        - Definição do Objeto e Quantitativos.
        - Fundamentação (Referência ao ETP).
        - Modelo de Execução e Gestão do Contrato.
        - Critérios de Pagamento e Medição.
        - Adequação Orçamentária.
        
        PERGUNTA: {question}
        CONTEXTO: {context}
        
        PARECER:
        Se faltar detalhe técnico (ex: prazo de garantia, SLA), aponte como falha.
        """
    
    # PROMPT DO PROJETO BÁSICO (OBRAS)
    else: 
        prompt_template = """
        Você é um Engenheiro Auditor do TCE.
        Analise o PROJETO BÁSICO (Obras) conforme Art. 6º, XXV da Lei 14.133 e Decreto 7.983/13.
        
        EXIGÊNCIAS:
        - Sondagens, Topografia e Estudos Geotécnicos (Obrigatório).
        - Orçamento Detalhado (Curva ABC + BDI discriminado).
        - Matriz de Riscos (Obras de Grande Vulto).
        - Cronograma Físico-Financeiro.
        
        PERGUNTA: {question}
        CONTEXTO: {context}
        
        PARECER:
        Se faltar BDI ou Cronograma, emita ALERTA VERMELHO de inexecutabilidade.
        """

    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key) 
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# --- 4. EXECUÇÃO DA AUDITORIA ---
def process_audit(vectorstore, uploaded_file, doc_type, questions_list):
    reader = PdfReader(uploaded_file)
    doc_text = ""
    for page in reader.pages:
        doc_text += page.extract_text()
    
    chain = get_specialized_chain(doc_type)
    results = []
    
    status = st.empty()
    progress = st.progress(0)
    
    full_audit_report = "" 
    
    for i, (titulo, pergunta_tecnica) in enumerate(questions_list):
        status.text(f"Auditando: {titulo}...")
        
        # BUSCA HÍBRIDA
        docs_lei = vectorstore.similarity_search(pergunta_tecnica, k=4)
        
        query_final = f"DOCUMENTO DO USUÁRIO (TEXTO REAL): {doc_text[:10000]} \n\n PERGUNTA: {pergunta_tecnica}"
        
        resp = chain.run(input_documents=docs_lei, question=query_final)
        
        results.append((titulo, resp))
        full_audit_report += f"\n- {titulo}: {resp}"
        progress.progress((i + 1) / len(questions_list))
    
    # GERAÇÃO DO RESUMO FINAL
    status.text("Gerando Relatório Conclusivo...")
    final_prompt = f"""
    Com base nas análises acima:
    {full_audit_report}
    
    Gere um RESUMO EXECUTIVO final listando APENAS:
    1. Itens Omitidos (O que falta).
    2. Alertas Vermelhos (Ilegalidades/Restrições).
    3. Conclusão: O documento está apto ou precisa de correção?
    """
    conclusao = chain.run(input_documents=docs_lei, question=final_prompt)
    results.append(("🏁 CONCLUSÃO FINAL DO AUDITOR", conclusao))
    
    status.empty()
    return results

# --- 5. TELA PRINCIPAL (LOGIN RESTAURADO) ---
def main():
    st.title("🏛️ AguiarGov - Auditor Fiscal (v5.2)")
    
    # BARRA LATERAL COM LOGIN
    with st.sidebar:
        st.header("🔐 Acesso Restrito")
        
        if not st.session_state['logged']:
            key = st.text_input("Digite sua Senha", type="password")
            if st.button("Entrar"):
                credits = check_login(key)
                if credits > -1:
                    st.session_state['logged'] = True
                    st.session_state['user_credits'] = credits
                    st.session_state['user_key'] = key
                    st.rerun() 
                else:
                    st.error("Senha inválida.")
        else:
            st.success(f"Logado: {st.session_state.get('user_key')}")
            st.info(f"Créditos: {st.session_state.get('user_credits')}")
            if st.button("Sair"):
                st.session_state['logged'] = False
                st.rerun()

    # CONTEÚDO PRINCIPAL (SÓ APARECE SE LOGADO)
    if st.session_state['logged']:
        # Carrega base
        if 'vectorstore' not in st.session_state or st.session_state['vectorstore'] is None:
            with st.spinner("Carregando Leis e Jurisprudência..."):
                vs, logs = load_knowledge_base()
                st.session_state['vectorstore'] = vs
                if vs is None: 
                    st.error(f"Erro: {logs}")
                elif st.session_state['user_key'] == "GUSTAVO_ADMIN":
                    with st.expander("🕵️ Logs do Sistema"):
                         for log in logs: st.write(log)
        
        vs = st.session_state.get('vectorstore')
        
        if vs:
            # MENU DE DOCUMENTOS
            modo = st.sidebar.radio("Selecione o Documento:", ["EDITAL", "ETP", "TR (Serviços)", "PROJETO BÁSICO (Obras)"])
            
            st.subheader(f"Auditoria de {modo}")
            uploaded = st.file_uploader(f"Suba o PDF do {modo}", type="pdf")
            
            if uploaded and st.button("AUDITAR AGORA"):
                
                if modo == "EDITAL":
                    qs = [
                        ("1. Objeto e Fundamentação", "O objeto está claro e sem direcionamento? A Lei 14.133 foi citada?"),
                        ("2. Habilitação (Restrições)", "Há exigências restritivas (sede local, capital > 10%, vistoria obrigatória)? Verifique Art. 62-70."),
                        ("3. Qualificação Técnica", "Os atestados exigidos são compatíveis e proporcionais?"),
                        ("4. Orçamento e Reajuste", "Há orçamento estimado ou referência ao TR? Há cláusula de reajuste obrigatória?"),
                        ("5. Prazos e Modos de Disputa", "Os prazos de publicidade e modo de disputa (aberto/fechado) estão corretos?")
                    ]
                    res = process_audit(vs, uploaded, "EDITAL", qs)
                
                elif modo == "ETP":
                    qs = [
                        ("1. Necessidade e PCA", "Descreve a necessidade pública e previsão no PCA (Inciso I e II)?"),
                        ("2. Requisitos e Quantidades", "Define requisitos e justifica quantidades com memória (III e IV)?"),
                        ("3. Levantamento de Mercado", "Analisou alternativas de mercado e justificou a solução (V e VII)?"),
                        ("4. Estimativa de Valor", "Tem estimativa de valor com preços unitários (VI)?"),
                        ("5. Parcelamento", "Justificou o parcelamento ou não (Inciso VIII)? Cite Súmula 247 TCU."),
                        ("6. Viabilidade", "Posicionamento conclusivo sobre viabilidade (XIII)?")
                    ]
                    res = process_audit(vs, uploaded, "ETP", qs)
                
                elif modo == "TR (Serviços)":
                    qs = [
                        ("1. Definição do Objeto", "Natureza, quantitativos e prazo (Art. 6, XXIII, a)?"),
                        ("2. Fundamentação", "Referência ao ETP correspondente (b)?"),
                        ("3. Gestão e Fiscalização", "Modelo de gestão e fiscalização do contrato (f)?"),
                        ("4. Pagamento e Medição", "Critérios claros de medição e pagamento (g)?"),
                        ("5. Seleção e Orçamento", "Critérios de seleção e adequação orçamentária (h, j)?")
                    ]
                    res = process_audit(vs, uploaded, "TR", qs)

                else: # OBRAS
                    qs = [
                        ("1. Engenharia (Sondagens)", "Há levantamentos topográficos e sondagens (Art. 6, XXV, a)?"),
                        ("2. Soluções Técnicas", "As soluções técnicas estão detalhadas (b)?"),
                        ("3. Cronograma e Métodos", "Há cronograma físico-financeiro e métodos construtivos?"),
                        ("4. Orçamento (BDI)", "Orçamento detalhado com BDI discriminado (Dec. 7.983)?")
                    ]
                    res = process_audit(vs, uploaded, "PB_OBRAS", qs)
                
                # EXIBIÇÃO
                st.markdown("---")
                st.header(f"📋 Relatório de Auditoria: {modo}")
                
                for tit, txt in res:
                    if "ALERTA VERMELHO" in txt or "🚨" in txt:
                        st.markdown(f"<div class='alert-box'><strong>{tit}</strong><br>{txt}</div>", unsafe_allow_html=True)
                    else:
                        with st.expander(f"✅ {tit}", expanded=True):
                            st.write(txt)

if __name__ == "__main__":
    main()
