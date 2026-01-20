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

# --- FUNÇÃO QUE LÊ OS PDFS AUTOMATICAMENTE ---
@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    text = ""
    data_folder = "data"
    subfolders = ["legislacao", "tcu_informativos", "tce_es_informativos", "doutrina_manuais"]
    files_processed = 0
    debug_log = [] # Lista para guardar os nomes dos arquivos encontrados
    
    if not os.path.exists(data_folder):
        return None, 0, ["ERRO: Pasta 'data' não encontrada na raiz."]

    # Varre as pastas e lê os PDFs
    for sub in subfolders:
        path = os.path.join(data_folder, sub)
        if os.path.exists(path):
            for filename in os.listdir(path):
                # CORREÇÃO: .lower() para ler .PDF maiúsculo também
                if filename.lower().endswith('.pdf'):
                    filepath = os.path.join(path, filename)
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
                            debug_log.append(f"✅ Lido: {sub}/{filename}")
                        else:
                            debug_log.append(f"⚠️ Vazio/Imagem: {sub}/{filename}")
                    except Exception as e:
                        debug_log.append(f"❌ Erro ao ler {filename}: {str(e)}")
        else:
            debug_log.append(f"⚠️ Pasta não encontrada: {sub}")
    
    if text == "":
        return None, 0, debug_log

    # Cria o "Cérebro"
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    
    # Gestão de Senha
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None, 0, ["ERRO CRÍTICO: Chave API não configurada."]
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    
    return vectorstore, files_processed, debug_log

# --- CÉREBRO JURÍDICO ---
def get_audit_chain():
    prompt_template = """
    Você é um Auditor Jurídico Especialista em Licitações (Brasil).
    Analise o trecho do edital abaixo.

    HIERARQUIA DE LEIS:
    1. Obras: Dec. 7.983/13 e Lei 14.133.
    2. Publicidade: Lei 12.232/10.
    3. Geral: Lei 14.133/21.

    Se encontrar erro, cite o Artigo.
    
    Contexto: {context}
    Pergunta: {question}
    
    Parecer:
    """
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = os.getenv("OPENAI_API_KEY")

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# --- LOGIN ---
def check_login(key):
    users = {
        "AMIGO_TESTE": 3,
        "PREFEITO_01": 3,
        "GUSTAVO_ADMIN": 99
    }
    return users.get(key, -1)

# --- TELA PRINCIPAL ---
def main():
    st.title("🏛️ AguiarGov - Auditor IA")
    
    with st.sidebar:
        st.header("Login")
        key = st.text_input("Senha de Acesso", type="password")
        if key:
            credits = check_login(key)
            if credits > -1:
                st.session_state['logged'] = True
                st.success(f"Logado! Créditos: {credits}")
            else:
                st.error("Senha Errada")
    
    if st.session_state.get('logged'):
        with st.spinner("Inicializando Base Jurídica..."):
            # Agora a função retorna também o LOG (lista do que aconteceu)
            vectorstore, qtd, logs = load_knowledge_base()
        
        # MOSTRAR LOGS SE DER ERRO (Debug)
        if qtd == 0:
            st.error("⚠️ Nenhum PDF foi processado. Veja o relatório abaixo:")
            with st.expander("Ver Relatório de Arquivos (Debug)"):
                for log in logs:
                    st.write(log)
        
        if vectorstore:
            st.success(f"Base Carregada: {qtd} arquivos lidos com sucesso.")
            uploaded_file = st.file_uploader("Arraste o Edital (PDF) aqui", type="pdf")
            
            if uploaded_file and st.button("Auditar Agora"):
                reader = PdfReader(uploaded_file)
                edital_text = ""
                for page in reader.pages:
                    edital_text += page.extract_text()
                
                questions = [
                    "Há exigência de Capital Social acima de 10%?",
                    "O critério de julgamento está correto para o objeto?",
                    "Há exigência de garantia contratual abusiva?"
                ]
                
                chain = get_audit_chain()
                st.write("---")
                for q in questions:
                    docs = vectorstore.similarity_search(q)
                    resp = chain.run(input_documents=docs, question=f"Edital: {edital_text[:3000]}... Pergunta: {q}")
                    st.markdown(f"#### 🧐 {q}")
                    st.write(resp)
                    st.write("---")

if __name__ == "__main__":
    main()
