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
        Você é um Auditor Especialista em Licitações e Jurisprudência (TCU/TCE).
        Analise o texto do EDITAL fornecido.
        Sua missão é cruzar as exigências do edital com a LEI 14.133/21 e a JURISPRUDÊNCIA fornecida.
        
        FOCO DA ANÁLISE: {question}
        
        DIRETRIZES OBRIGATÓRIAS:
        1. Se encontrar irregularidade, cite o Artigo da Lei.
        2. CITE A FONTE JURISPRUDENCIAL se houver no contexto (Ex: "Conforme Acórdão TCU nº X", "Segundo Súmula Y").
        3. Seja técnico e direto.
        
        Contexto: {context}
        PARECER TÉCNICO:
        """

    elif doc_type == "ETP":
        prompt_template = """
        Você é um Auditor Focado em Planejamento.
        Analise o ETP à luz do Art. 18 da Lei 14.133/21 e das orientações dos Manuais de Planejamento (TCU/TCE).
        
        FOCO DA ANÁLISE: {question}
        
        DIRETRIZES:
        - Verifique os incisos do Art. 18.
        - Se o texto contrariar algum entendimento consolidado, aponte a divergência.
        
        Contexto: {context}
        PARECER SOBRE O ETP:
        """

    else: # TR
        prompt_template = """
        Você é um Auditor Técnico.
        Analise o Termo de Referência (TR).
        
        FOCO DA ANÁLISE: {question}
        
        DIRETRIZES:
        - Valide a definição do objeto (Art. 6º, XXIII).
        - Verifique se há restrição indevida.
        - Use a jurisprudência fornecida para embasar.
        
        Contexto: {context}
        PARECER SOBRE O TR:
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

# --- 4. FUNÇÃO QUE RODA A ANÁLISE ---
def run_analysis(vectorstore, uploaded_file, doc_type, questions):
    reader = PdfReader(uploaded_file)
    doc_text = ""
    for page in reader.pages:
        doc_text += page.extract_text()
    
    chain = get_specialized_chain(doc_type)
    
    # Título discreto no relatório
    st.markdown(f"### 📋 Resultado da Análise ({doc_type})")
    progress_bar = st.progress(0)
    
    for i, q in enumerate(questions):
        docs = vectorstore.similarity_search(q)
        resp = chain.run(input_documents=docs, question=f"Texto do Documento: {doc_text[:6000]}... TAREFA: {q}")
        
        with st.chat_message("assistant"):
            # Aqui mantemos a pergunta visível APENAS no relatório final
            st.markdown(f"**Item Analisado:** {q}")
            st.write(resp)
        
        progress_bar.progress((i + 1) / len(questions))
    
    st.success(f"✅ Análise Finalizada.")

# --- 5. TELA PRINCIPAL (LIMPA/BLACK BOX) ---
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
        with st.spinner("Inicializando o sistema..."):
            vectorstore, qtd, logs = load_knowledge_base()
        
        if st.session_state.get('user_key') == "GUSTAVO_ADMIN" and qtd > 0:
             with st.expander("🕵️ Logs do Admin"):
                for log in logs: st.write(log)
        
        if vectorstore:
            # --- ABAS LIMPAS (SEM EXPLICAÇÃO TÉCNICA) ---
            tab1, tab2, tab3 = st.tabs(["📄 EDITAL", "📘 ETP", "📋 TR / P. BÁSICO"])
            
            # --- ABA 1: EDITAL ---
            with tab1:
                # Sem st.info explicando o que faz
                file_edital = st.file_uploader("Selecione o arquivo PDF do Edital", type="pdf", key="u1")
                if file_edital and st.button("AUDITAR ARQUIVO (1 Crédito)", key="b1"):
                    questions = [
                        "Verifique a MODALIDADE e o CRITÉRIO DE JULGAMENTO. Estão adequados ao objeto? (Art. 28 e 33)",
                        "Analise os REQUISITOS DE HABILITAÇÃO (Jurídica, Fiscal, Técnica, Econômica). Há excessos ou restrições? (Art. 62 a 70)",
                        "Verifique os PRAZOS DE PUBLICAÇÃO e de IMPUGNAÇÃO. Estão corretos? (Art. 55 e 164)"
                    ]
                    run_analysis(vectorstore, file_edital, "EDITAL", questions)

            # --- ABA 2: ETP ---
            with tab2:
                file_etp = st.file_uploader("Selecione o arquivo PDF do ETP", type="pdf", key="u2")
                if file_etp and st.button("AUDITAR ARQUIVO (1 Crédito)", key="b2"):
                    questions = [
                        "O ETP descreve a NECESSIDADE da contratação de forma clara? (Inciso I)",
                        "Houve LEVANTAMENTO DE MERCADO e análise de alternativas? (Inciso III)",
                        "Há ESTIMATIVA DO VALOR e adequação orçamentária? (Inciso VI e VII)",
                        "A ESCOLHA DA SOLUÇÃO foi justificada técnica e economicamente? (Inciso VIII)"
                    ]
                    run_analysis(vectorstore, file_etp, "ETP", questions)

            # --- ABA 3: TR ---
            with tab3:
                file_tr = st.file_uploader("Selecione o arquivo PDF do TR", type="pdf", key="u3")
                if file_tr and st.button("AUDITAR ARQUIVO (1 Crédito)", key="b3"):
                    questions = [
                        "A definição do OBJETO é precisa, suficiente e clara? Há vedação de marca? (Inciso XXIII, 'a')",
                        "O MODELO DE EXECUÇÃO do objeto está claro? (Inciso XXIII, 'e')",
                        "Os CRITÉRIOS DE MEDIÇÃO E PAGAMENTO estão definidos objetivamente? (Inciso XXIII, 'h')",
                        "Há previsão de FISCALIZAÇÃO e critérios de recebimento? (Inciso XXIII, 'g')"
                    ]
                    run_analysis(vectorstore, file_tr, "TR", questions)

if __name__ == "__main__":
    main()
