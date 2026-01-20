import streamlit as st
import os
from PyPDF2 import PdfReader
# VERSÃO ESTÁVEL LANGCHAIN 0.1
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Configuração da Página (Títulos e Ícone)
st.set_page_config(page_title="LicitaGov - Auditor IA", page_icon="⚖️", layout="wide")

# --- FUNÇÃO QUE LÊ OS PDFS (MODO SILENCIOSO) ---
@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    text = ""
    data_folder = "data"
    files_processed = 0
    debug_log = [] 
    
    if not os.path.exists(data_folder):
        return None, 0, ["ERRO: Pasta 'data' não encontrada na raiz."]

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
                    else:
                        debug_log.append(f"⚠️ Arquivo Vazio/Imagem: {filename}")
                except Exception as e:
                    debug_log.append(f"❌ Erro ao ler {filename}: {str(e)}")
    
    if text == "":
        return None, 0, debug_log

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    
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

# --- LOGIN E CRÉDITOS ---
def check_login(key):
    # Dicionário de Usuários
    # Dica: Adicione seus amigos aqui com 3 créditos
    users = {
        "AMIGO_TESTE": 3,
        "PREFEITURA_X": 3,
        "GUSTAVO_ADMIN": 99  # Só esse usuário vê os logs
    }
    return users.get(key, -1)

# --- TELA PRINCIPAL ---
def main():
    st.title("🏛️ AguiarGov - Auditor IA")
    st.markdown("---")
    
    with st.sidebar:
        st.header("🔐 Área Restrita")
        key = st.text_input("Digite sua Senha de Acesso", type="password")
        if key:
            credits = check_login(key)
            if credits > -1:
                st.session_state['logged'] = True
                st.session_state['user_key'] = key # Salva quem está logado
                st.success(f"Bem-vindo! Créditos de Análise: {credits}")
            else:
                st.error("Senha não encontrada.")
    
    if st.session_state.get('logged'):
        with st.spinner("🤖 O Robô está estudando as leis... aguarde um momento."):
            vectorstore, qtd, logs = load_knowledge_base()
        
        # --- SEGREDO: SÓ O ADMIN VÊ OS LOGS ---
        # Se a chave for GUSTAVO_ADMIN, mostra a lista. Se for AMIGO_TESTE, esconde.
        current_user = st.session_state.get('user_key')
        
        if current_user == "GUSTAVO_ADMIN" and qtd > 0:
            with st.expander(f"🕵️ Painel do Admin ({qtd} arquivos carregados)"):
                for log in logs:
                    st.write(log)
        
        # Se deu erro geral (0 arquivos), avisa todo mundo
        elif qtd == 0:
            st.error("⚠️ Erro no sistema: Base de dados vazia. Contate o suporte.")
        
        # --- ÁREA DO CLIENTE ---
        if vectorstore:
            st.markdown("### 📂 Upload do Edital")
            st.info("O sistema aceita arquivos PDF de até 200MB.")
            
            # Botão traduzido no Rótulo
            uploaded_file = st.file_uploader("Clique abaixo para selecionar o arquivo PDF do seu computador", type="pdf")
            
            if uploaded_file:
                st.success("Arquivo recebido! Clique no botão abaixo para iniciar.")
                if st.button("🚀 AUDITAR AGORA (Gastar 1 Crédito)"):
                    reader = PdfReader(uploaded_file)
                    edital_text = ""
                    for page in reader.pages:
                        edital_text += page.extract_text()
                    
                    # As perguntas que o robô vai responder
                    questions = [
                        "Há exigência de Capital Social acima de 10%?",
                        "O critério de julgamento está correto para o objeto?",
                        "Há exigência de garantia contratual abusiva?"
                    ]
                    
                    chain = get_audit_chain()
                    st.write("---")
                    st.subheader("📋 Relatório da Auditoria")
                    
                    progress_bar = st.progress(0)
                    
                    for i, q in enumerate(questions):
                        docs = vectorstore.similarity_search(q)
                        resp = chain.run(input_documents=docs, question=f"Edital: {edital_text[:3000]}... Pergunta: {q}")
                        
                        with st.chat_message("assistant"):
                            st.markdown(f"**Pergunta {i+1}:** {q}")
                            st.write(resp)
                        
                        # Atualiza a barra de progresso
                        progress_bar.progress((i + 1) / len(questions))
                        
                    st.success("✅ Auditoria Finalizada!")

if __name__ == "__main__":
    main()
