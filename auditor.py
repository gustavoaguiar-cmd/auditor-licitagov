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

# --- 2. CÉREBRO ESPECIALISTA ---
def get_specialized_chain(doc_type):
    
    if doc_type == "EDITAL":
        prompt_template = """
        Você é um Auditor Rigoroso de Licitações (Controle Externo).
        Analise o EDITAL fornecido.
        
        REQUISITOS LEGAIS:
        1. Lei 14.133/21 (Art. 25).
        2. Jurisprudência: Prejulgados do TCE/ES e Acórdãos do TCU.
        
        PERGUNTA DA AUDITORIA: {question}
        
        REGRAS CRÍTICAS:
        - Responda estritamente com base no texto do documento.
        - Se não encontrar o item, diga: "OMISSÃO: Item não localizado no edital."
        - Cite o artigo da lei violado ou atendido.
        - NÃO COLOQUE ASSINATURA.
        
        Contexto Legal: {context}
        PARECER TÉCNICO:
        """

    elif doc_type == "ETP":
        prompt_template = """
        Você é um Auditor de Planejamento.
        Analise o ETP com base RIGOROSA no Art. 18, §1º da Lei 14.133/21.
        
        PERGUNTA DA AUDITORIA: {question}
        
        REGRAS CRÍTICAS:
        - Verifique os incisos do Art. 18.
        - Se o texto não trouxer a informação explicita, aponte como OMISSÃO.
        - NÃO COLOQUE ASSINATURA.
        
        Contexto Legal: {context}
        PARECER SOBRE O ETP:
        """

    elif doc_type == "TR_SERVICO": # NOVO: Só para Serviços Comuns
        prompt_template = """
        Você é um Auditor de Licitações.
        Analise o Termo de Referência (TR) para BENS OU SERVIÇOS COMUNS.
        Base Legal: Art. 6º, XXIII da Lei 14.133/21.
        
        PERGUNTA DA AUDITORIA: {question}
        
        REGRAS CRÍTICAS:
        - Busque a evidência APENAS no texto.
        - Não exija itens de engenharia (como BDI ou Projeto Básico) pois é um Serviço Comum.
        - Se não encontrar, diga: "OMISSÃO".
        - NÃO COLOQUE ASSINATURA.
        
        Contexto Legal: {context}
        PARECER SOBRE O TR:
        """
        
    else: # PROJETO BÁSICO (OBRAS)
        prompt_template = """
        Você é um Auditor de Engenharia (Obras Públicas).
        Analise o PROJETO BÁSICO DE ENGENHARIA.
        Base Legal: Lei 14.133/21 (Art. 6º, XXV) e Decreto 7.983/13.
        
        PERGUNTA DA AUDITORIA: {question}
        
        REGRAS CRÍTICAS:
        - Exija rigorosamente BDI, Curva ABC, Cronograma e Sondagens se aplicável.
        - Verifique alinhamento com SINAPI/SICRO.
        - NÃO COLOQUE ASSINATURA.
        
        Contexto Legal: {context}
        PARECER SOBRE O PROJETO BÁSICO:
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
        return [("Erro de Leitura", "O arquivo PDF parece ser uma imagem ou está vazio.")]

    chain = get_specialized_chain(doc_type)
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (titulo_bonito, prompt_tecnico) in enumerate(questions_list):
        status_text.text(f"Auditando: {titulo_bonito}...")
        docs = vectorstore.similarity_search(prompt_tecnico)
        
        resp = chain.run(input_documents=docs, question=f"Texto do Documento (FONTE DA VERDADE): {doc_text[:7000]}... TAREFA: {prompt_tecnico}")
        
        results.append((titulo_bonito, resp))
        progress_bar.progress((i + 1) / len(questions_list))
    
    status_text.text("Auditoria Concluída!")
    return results

# --- 5. EXIBIÇÃO ---
def display_results(results_list, doc_type):
    if results_list:
        st.markdown(f"### 📋 Relatório: {doc_type}")
        for titulo, resposta in results_list:
            with st.chat_message("assistant"):
                st.markdown(f"**{titulo}**")
                st.write(resposta)

# --- 6. TELA PRINCIPAL ---
def main():
    st.title("🏛️ AguiarGov - Auditor IA")
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
        with st.spinner("Carregando Base Jurídica..."):
            vectorstore, qtd, logs = load_knowledge_base()
        
        if st.session_state.get('user_key') == "GUSTAVO_ADMIN" and qtd > 0:
             with st.expander("🕵️ Logs do Admin"):
                for log in logs: st.write(log)
        
        if vectorstore:
            tab1, tab2, tab3 = st.tabs(["📄 EDITAL", "📘 ETP", "📋 TR / P. BÁSICO"])
            
            # --- ABA 1: EDITAL ---
            with tab1:
                file_edital = st.file_uploader("Suba o EDITAL", type="pdf", key="u1")
                if file_edital and st.button("AUDITAR EDITAL", key="b1"):
                    questions = [
                        ("1. Objeto e Regras (Art. 25)", "O edital contém objeto, julgamento, habilitação e recursos conforme Art. 25?"),
                        ("2. Minuta e Divulgação", "Foi utilizada minuta padronizada e prevista divulgação em sítio eletrônico?"),
                        ("3. Orçamento e Reajuste", "Há orçamento estimado e previsão OBRIGATÓRIA de índice de reajustamento?"),
                        ("4. Matriz de Riscos", "Há previsão de Matriz de Riscos ou Programa de Integridade (se aplicável)?"),
                        ("5. Habilitação", "A habilitação respeita a Lei 14.133 (Art 62 a 70)?")
                    ]
                    st.session_state['result_edital'] = process_audit(vectorstore, file_edital, "EDITAL", questions)
                
                if st.session_state['result_edital']:
                    display_results(st.session_state['result_edital'], "EDITAL")

            # --- ABA 2: ETP ---
            with tab2:
                file_etp = st.file_uploader("Suba o ETP", type="pdf", key="u2")
                if file_etp and st.button("AUDITAR ETP", key="b2"):
                    questions = [
                        ("1. Necessidade (Inciso I)", "Descrição da necessidade sob a perspectiva do interesse público?"),
                        ("2. Plano de Contratações (Inciso II)", "Previsão no Plano de Contratações Anual?"),
                        ("3. Requisitos (Inciso III)", "Definição dos requisitos da contratação?"),
                        ("4. Quantidades (Inciso IV)", "Estimativas das quantidades com memórias de cálculo?"),
                        ("5. Mercado (Inciso V)", "Levantamento de mercado e análise de alternativas?"),
                        ("6. Valor (Inciso VI)", "Estimativa do valor com preços unitários?"),
                        ("7. Solução (Inciso VII)", "Descrição da solução como um todo?"),
                        ("8. Parcelamento (Inciso VIII)", "Justificativas para o parcelamento ou não?"),
                        ("9. Resultados (Inciso IX)", "Demonstrativo dos resultados pretendidos?"),
                        ("10. Providências (Inciso X)", "Providências prévias ao contrato?"),
                        ("11. Correlatas (Inciso XI)", "Contratações correlatas/interdependentes?"),
                        ("12. Ambiental (Inciso XII)", "Impactos ambientais e medidas mitigadoras?"),
                        ("13. Viabilidade (Inciso XIII)", "Posicionamento conclusivo sobre viabilidade?")
                    ]
                    st.session_state['result_etp'] = process_audit(vectorstore, file_etp, "ETP", questions)
                
                if st.session_state['result_etp']:
                    display_results(st.session_state['result_etp'], "ETP")

            # --- ABA 3: TR ou PROJETO BÁSICO (SELETOR) ---
            with tab3:
                # AQUI ESTÁ A SOLUÇÃO DA CONFUSÃO
                st.info("Selecione o tipo de objeto para a auditoria correta:")
                tipo_doc = st.radio("O que você vai auditar?", 
                                    ["Termo de Referência (Bens e Serviços Comuns)", 
                                     "Projeto Básico (Obras e Engenharia)"])
                
                file_tr = st.file_uploader("Suba o Arquivo (TR ou PB)", type="pdf", key="u3")
                
                if file_tr and st.button("AUDITAR TR/PB", key="b3"):
                    
                    if tipo_doc == "Termo de Referência (Bens e Serviços Comuns)":
                        # LISTA LEVE (Serviços)
                        questions = [
                            ("1. Definição do Objeto", "Definição do objeto, natureza, quantitativos e prazo?"),
                            ("2. Fundamentação", "Fundamentação da contratação com referência ao ETP?"),
                            ("3. Solução", "Descrição da solução como um todo?"),
                            ("4. Execução", "Definição do modelo de execução do objeto?"),
                            ("5. Gestão", "Modelo de gestão do contrato?"),
                            ("6. Pagamento", "Critérios de medição e pagamento?"),
                            ("7. Seleção", "Forma e critérios de seleção do fornecedor?"),
                            ("8. Estimativa", "Estimativas de valor e memórias de cálculo?"),
                            ("9. Orçamento", "Declaração de adequação orçamentária?")
                        ]
                        # Usa o Cérebro de Serviço
                        st.session_state['result_tr'] = process_audit(vectorstore, file_tr, "TR_SERVICO", questions)
                    
                    else:
                        # LISTA PESADA (Obras - Art. 6 XXV)
                        questions = [
                            ("1. Levantamentos Técnicos", "Contém levantamentos topográficos, sondagens e estudos geotécnicos?"),
                            ("2. Soluções Técnicas", "Soluções técnicas globais e localizadas detalhadas?"),
                            ("3. Especificações", "Identificação de serviços, materiais e equipamentos com especificações?"),
                            ("4. Cronograma/Métodos", "Definição de métodos construtivos e cronograma?"),
                            ("5. Orçamento Detalhado (Dec. 7.983)", "Orçamento detalhado do custo global com BDI e Encargos Sociais discriminados?"),
                            ("6. Licenciamento Ambiental", "O projeto trata do licenciamento e impacto ambiental do empreendimento?"),
                            ("7. ART/RRT", "Há anotação de responsabilidade técnica (ART) dos projetistas?")
                        ]
                        # Usa o Cérebro de Obras
                        st.session_state['result_tr'] = process_audit(vectorstore, file_tr, "PB_OBRAS", questions)
                
                if st.session_state['result_tr']:
                    display_results(st.session_state['result_tr'], "TR/PB")

if __name__ == "__main__":
    main()
