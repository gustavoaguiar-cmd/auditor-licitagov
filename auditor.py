import streamlit as st
import os
from PyPDF2 import PdfReader
# VERSÃO ESTÁVEL LANGCHAIN 0.1
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Configuração da Página
st.set_page_config(page_title="LicitaGov - Auditor IA", page_icon="⚖️", layout="wide")

# --- INICIALIZAR MEMÓRIA (SESSION STATE) ---
if 'result_edital' not in st.session_state:
    st.session_state['result_edital'] = None
if 'result_etp' not in st.session_state:
    st.session_state['result_etp'] = None
if 'result_tr' not in st.session_state:
    st.session_state['result_tr'] = None

# --- 1. CARREGAMENTO DA BASE JURÍDICA ---
@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    text = ""
    data_folder = "data"
    files_processed = 0
    debug_log = [] 
    
    if not os.path.exists(data_folder):
        return None, 0, ["ERRO: Pasta 'data' não encontrada."]

    for root, dirs, files in os.walk(data_folder):
        for filename in files:
            if filename.lower().endswith('.pdf'):
                filepath = os.path.join(root, filename)
                try:
                    pdf_reader = PdfReader(filepath)
                    file_text = ""
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            file_text += page_text
                    
                    if file_text:
                        text += file_text
                        files_processed += 1
                        folder_name = os.path.basename(root)
                        debug_log.append(f"✅ Lido ({folder_name}): {filename}")
                except Exception:
                    continue
    
    if text == "":
        return None, 0, debug_log

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None, 0, ["ERRO: Chave API ausente."]
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore, files_processed, debug_log

# --- 2. CÉREBRO ESPECIALISTA (ATUALIZADO COM LEIS ESPECÍFICAS) ---
def get_specialized_chain(doc_type):
    
    # PROMPT DO EDITAL (Art. 25 + Leis Específicas)
    if doc_type == "EDITAL":
        prompt_template = """
        Você é um Auditor Rigoroso de Licitações (Controle Externo).
        Analise o EDITAL fornecido.
        
        REQUISITOS LEGAIS OBRIGATÓRIOS:
        1. Lei 14.133/21 (Art. 25 e seguintes).
        2. Se for OBRAS: Decreto 7.983/13 (BDI, Sinapi) e Art. 23 da Lei 14.133.
        3. Se for PUBLICIDADE: Lei 12.232/10.
        4. Jurisprudência: Siga Prejulgados do TCE/ES e Acórdãos do TCU.
        
        PERGUNTA DA AUDITORIA: {question}
        
        REGRAS CRÍTICAS (ANTI-ALUCINAÇÃO):
        - Responda estritamente com base no texto do documento.
        - Se não encontrar o item, diga: "OMISSÃO: Item não localizado no edital."
        - Cite o artigo da lei violado ou atendido.
        - NÃO COLOQUE ASSINATURA.
        
        Contexto Legal: {context}
        PARECER TÉCNICO:
        """

    # PROMPT DO ETP (Art. 18, §1º)
    elif doc_type == "ETP":
        prompt_template = """
        Você é um Auditor de Planejamento.
        Analise o ETP com base RIGOROSA no Art. 18, §1º da Lei 14.133/21.
        
        PERGUNTA DA AUDITORIA: {question}
        
        REGRAS CRÍTICAS:
        - Verifique se o documento contém os elementos dos incisos I a XIII do Art. 18.
        - Se o texto não trouxer a informação explicita, aponte como OMISSÃO.
        - Use a jurisprudência do TCU/TCE-ES para embasar a profundidade da resposta.
        - NÃO COLOQUE ASSINATURA.
        
        Contexto Legal: {context}
        PARECER SOBRE O ETP:
        """

    # PROMPT DO TR / PROJETO BÁSICO (Art. 6º)
    else: # TR
        prompt_template = """
        Você é um Auditor Técnico de Engenharia e Serviços.
        Analise o Termo de Referência (TR) ou Projeto Básico (PB).
        
        REQUISITOS:
        - TR: Art. 6º, XXIII da Lei 14.133/21.
        - Projeto Básico (Obras): Art. 6º, XXV da Lei 14.133/21 e Dec. 7.983.
        
        PERGUNTA DA AUDITORIA: {question}
        
        REGRAS CRÍTICAS:
        - Busque a evidência APENAS no texto do documento.
        - Se for obra, verifique sondagens, orçamento detalhado e BDI.
        - Se não encontrar, diga: "OMISSÃO: Tópico não localizado."
        - NÃO COLOQUE ASSINATURA.
        
        Contexto Legal: {context}
        PARECER SOBRE O TR/PB:
        """

    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = os.getenv("OPENAI_API_KEY")

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# --- 3. LOGIN ---
def check_login(key):
    users = {
        "AMIGO_TESTE": 3,
        "PREFEITURA_X": 10,
        "GUSTAVO_ADMIN": 99
    }
    return users.get(key, -1)

# --- 4. PROCESSAMENTO ---
def process_audit(vectorstore, uploaded_file, doc_type, questions_list):
    reader = PdfReader(uploaded_file)
    doc_text = ""
    for page in reader.pages:
        doc_text += page.extract_text()
    
    if len(doc_text) < 50:
        return [("Erro de Leitura", "O arquivo PDF parece ser uma imagem ou está vazio. Não foi possível ler.")]

    chain = get_specialized_chain(doc_type)
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (titulo_bonito, prompt_tecnico) in enumerate(questions_list):
        status_text.text(f"Auditando: {titulo_bonito}...")
        docs = vectorstore.similarity_search(prompt_tecnico)
        
        # Injeção de Contexto Rigoroso
        resp = chain.run(input_documents=docs, question=f"Texto do Documento (FONTE DA VERDADE): {doc_text[:7000]}... TAREFA: {prompt_tecnico}")
        
        results.append((titulo_bonito, resp))
        progress_bar.progress((i + 1) / len(questions_list))
    
    status_text.text("Auditoria Concluída!")
    return results

# --- 5. EXIBIÇÃO ---
def display_results(results_list, doc_type):
    if results_list:
        st.markdown(f"### 📋 Relatório de Auditoria ({doc_type})")
        for titulo, resposta in results_list:
            with st.chat_message("assistant"):
                st.markdown(f"**{titulo}**")
                st.write(resposta)

# --- 6. TELA PRINCIPAL ---
def main():
    st.title("🏛️ AguiarGov - Auditor IA (Compliance 14.133)")
    st.markdown("---")
    
    with st.sidebar:
        st.header("🔐 Acesso")
        key = st.text_input("Senha", type="password")
        if key:
            credits = check_login(key)
            if credits > -1:
                st.session_state['logged'] = True
                st.session_state['user_key'] = key
                st.success(f"Logado. Créditos: {credits}")
            else:
                st.error("Senha inválida.")
    
    if st.session_state.get('logged'):
        with st.spinner("Carregando Base Jurídica e Leis..."):
            vectorstore, qtd, logs = load_knowledge_base()
        
        if st.session_state.get('user_key') == "GUSTAVO_ADMIN" and qtd > 0:
             with st.expander("🕵️ Logs do Admin"):
                for log in logs: st.write(log)
        
        if vectorstore:
            tab1, tab2, tab3 = st.tabs(["📄 EDITAL", "📘 ETP", "📋 TR / P. BÁSICO"])
            
            # --- ABA 1: EDITAL (Art. 25 + Leis Específicas) ---
            with tab1:
                file_edital = st.file_uploader("Suba o EDITAL", type="pdf", key="u1")
                if file_edital and st.button("AUDITAR EDITAL (1 Crédito)", key="b1"):
                    questions = [
                        ("1. Objeto e Regras Gerais (Art. 25)", "O edital contém objeto, regras de convocação, julgamento, habilitação, recursos e penalidades conforme Art. 25?"),
                        ("2. Minuta Padronizada e Divulgação", "Foi utilizada minuta padronizada (§1º) e prevista divulgação em sítio eletrônico (§3º)?"),
                        ("3. Orçamento e Reajuste (Art. 25, §7º)", "Há orçamento estimado e previsão OBRIGATÓRIA de índice de reajustamento de preços?"),
                        ("4. Matriz de Riscos e Integridade", "Há previsão de Matriz de Riscos ou Programa de Integridade (se for grande vulto)?"),
                        ("5. Critério de Julgamento e Habilitação", "O critério de julgamento e a habilitação respeitam a Lei 14.133?"),
                        ("6. Obras (Dec. 7983) ou Publicidade (12.232)", "Se for OBRA: Respeita o Dec. 7.983 (Sinapi/BDI)? Se for PUBLICIDADE: Respeita Lei 12.232?")
                    ]
                    st.session_state['result_edital'] = process_audit(vectorstore, file_edital, "EDITAL", questions)
                
                if st.session_state['result_edital']:
                    display_results(st.session_state['result_edital'], "EDITAL")

            # --- ABA 2: ETP (Art. 18, §1º - COMPLETO) ---
            with tab2:
                file_etp = st.file_uploader("Suba o ETP", type="pdf", key="u2")
                if file_etp and st.button("AUDITAR ETP (1 Crédito)", key="b2"):
                    questions = [
                        ("1. Necessidade (Inciso I)", "Descrição da necessidade sob a perspectiva do interesse público?"),
                        ("2. Plano de Contratações (Inciso II)", "Demonstração da previsão no Plano de Contratações Anual?"),
                        ("3. Requisitos (Inciso III)", "Definição dos requisitos da contratação?"),
                        ("4. Quantidades e Memória (Inciso IV)", "Estimativas das quantidades acompanhadas das memórias de cálculo?"),
                        ("5. Levantamento de Mercado (Inciso V)", "Levantamento de mercado, análise de alternativas e justificativa da escolha?"),
                        ("6. Estimativa de Valor (Inciso VI)", "Estimativa do valor com preços unitários e memórias de cálculo?"),
                        ("7. Descrição da Solução (Inciso VII)", "Descrição da solução como um todo, inclusive manutenção/assistência?"),
                        ("8. Parcelamento (Inciso VIII)", "Justificativas para o parcelamento ou não da contratação?"),
                        ("9. Resultados Pretendidos (Inciso IX)", "Demonstrativo dos resultados pretendidos (economicidade/eficiência)?"),
                        ("10. Providências Prévias (Inciso X)", "Providências a serem adotadas antes do contrato (capacitação/fiscalização)?"),
                        ("11. Contratações Correlatas (Inciso XI)", "Análise de contratações correlatas e/ou interdependentes?"),
                        ("12. Impactos Ambientais (Inciso XII)", "Descrição de impactos ambientais e medidas mitigadoras?"),
                        ("13. Viabilidade (Inciso XIII)", "Posicionamento conclusivo sobre a adequação e viabilidade da contratação?")
                    ]
                    st.session_state['result_etp'] = process_audit(vectorstore, file_etp, "ETP", questions)
                
                if st.session_state['result_etp']:
                    display_results(st.session_state['result_etp'], "ETP")

            # --- ABA 3: TR (Art. 6, XXIII) e PB (Art. 6, XXV) ---
            with tab3:
                file_tr = st.file_uploader("Suba TR ou PROJETO BÁSICO", type="pdf", key="u3")
                if file_tr and st.button("AUDITAR TR/PB (1 Crédito)", key="b3"):
                    questions = [
                        ("1. Definição do Objeto (Alínea 'a')", "Definição do objeto, natureza, quantitativos e prazo (com possibilidade de prorrogação)?"),
                        ("2. Fundamentação e ETP (Alínea 'b')", "Fundamentação da contratação com referência aos Estudos Técnicos Preliminares?"),
                        ("3. Solução e Ciclo de Vida (Alínea 'c')", "Descrição da solução como um todo, considerado o ciclo de vida?"),
                        ("4. Modelo de Execução (Alínea 'e')", "Definição de como o contrato deverá produzir os resultados (Modelo de Execução)?"),
                        ("5. Gestão e Fiscalização (Alínea 'f')", "Modelo de gestão do contrato (como será fiscalizado)?"),
                        ("6. Medição e Pagamento (Alínea 'g')", "Critérios objetivos de medição e de pagamento?"),
                        ("7. Seleção do Fornecedor (Alínea 'h')", "Forma e critérios de seleção do fornecedor?"),
                        ("8. Estimativa de Valor (Alínea 'i')", "Estimativas de valor, preços unitários, memórias de cálculo e parâmetros utilizados?"),
                        ("9. Adequação Orçamentária (Alínea 'j')", "Declaração de adequação orçamentária?"),
                        ("10. SE FOR OBRA (Projeto Básico - Art. 6, XXV)", "Contém levantamentos topográficos, sondagens, orçamento detalhado e BDI (Dec 7983)?")
                    ]
                    st.session_state['result_tr'] = process_audit(vectorstore, file_tr, "TR", questions)
                
                if st.session_state['result_tr']:
                    display_results(st.session_state['result_tr'], "TR")

if __name__ == "__main__":
    main()
