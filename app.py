app.py
import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Configuração da Página
st.set_page_config(page_title="LicitaGov - Auditor IA", page_icon="⚖️", layout="wide")

# Carrega variáveis locais (para quando você testar no PC, se quiser)
load_dotenv()

# --- FUNÇÃO QUE LÊ OS PDFS AUTOMATICAMENTE ---
@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    text = ""
    # O Streamlit vai procurar a pasta 'data' junto com o código
    data_folder = "data"
    
    subfolders = ["legislacao", "tcu_informativos", "tce_es_informativos", "doutrina_manuais"]
    files_processed = 0
    
    if not os.path.exists(data_folder):
        return None, 0

    # Varre as pastas e lê os PDFs
    for sub in subfolders:
        path = os.path.join(data_folder, sub)
        if os.path.exists(path):
            for filename in os.listdir(path):
                if filename.endswith('.pdf'):
                    filepath = os.path.join(path, filename)
                    try:
                        pdf_reader = PdfReader(filepath)
                        for page in pdf_reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text
                        files_processed += 1
                    except:
                        continue 
    
    if text == "":
        return None, 0

    # Cria o "Cérebro"
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    
    # Pega a senha (do Site ou do Arquivo Local)
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    
    return vectorstore, files_processed

# --- CÉREBRO JURÍDICO ---
def get_audit_chain():
    prompt_template = """
    Você é um Auditor Jurídico Especialista em Licitações (Brasil).
    Analise o trecho do edital abaixo.

    HIERARQUIA DE LEIS (Siga estritamente):
    1. Obras: Aplique Dec. 7.983/13 e Lei 14.133.
    2. Publicidade: Aplique Lei 12.232/10.
    3. Geral: Aplique Lei 14.133/21.

    Se encontrar erro, cite o Artigo. Se estiver certo, diga "Nenhuma irregularidade".
    
    Contexto Legal: {context}
    Pergunta/Edital: {question}
    
    Parecer:
    """
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# --- SENHAS DE ACESSO (Login) ---
def check_login(key):
    # Aqui você define as senhas dos seus amigos
    users = {
        "AMIGO_TESTE": 3,
        "PREFEITO_01": 3,
        "GUSTAVO_ADMIN": 99
    }
    return users.get(key, -1)

# --- TELA DO SISTEMA ---
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
        with st.spinner("Lendo PDFs e criando inteligência..."):
            vectorstore, qtd = load_knowledge_base()
        
        if vectorstore:
            st.success(f"Base Carregada: {qtd} arquivos jurídicos lidos.")
            uploaded_file = st.file_uploader("Arraste o Edital (PDF) aqui", type="pdf")
            
            if uploaded_file and st.button("Auditar Agora"):
                reader = PdfReader(uploaded_file)
                edital_text = ""
                for page in reader.pages:
                    edital_text += page.extract_text()
                
                # Perguntas Automáticas
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
        else:
            st.warning("Nenhum PDF encontrado na pasta data.")

if __name__ == "__main__":
    main()