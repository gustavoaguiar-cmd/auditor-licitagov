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

# --- 2. CÉREBRO ESPECIALISTA (MODO CÉTICO/ANTI-ALUCINAÇÃO) ---
def get_specialized_chain(doc_type):
    
    # REGRAS DE OURO ADICIONADAS NOS PROMPTS:
    # "BASEIE-SE APENAS NO TEXTO DO DOCUMENTO."
    # "Se a informação não estiver explicita, diga: NÃO CONSTA NO DOCUMENTO."
    
    if doc_type == "EDITAL":
        prompt_template = """
        Você é um Auditor Rigoroso de Licitações.
        Analise o texto do DOCUMENTO UPLOADED (Edital) fornecido abaixo.
        
        Sua missão é verificar se o texto do documento contém as exigências da LEI 14.133/21.
        
        PERGUNTA DA AUDITORIA: {question}
        
        REGRAS CRÍTICAS DE RESPOSTA (Anti-Alucinação):
        1. Responda APENAS com base no que está escrito no "Texto do Documento".
        2. Se o documento for um Boleto, Receita, ou texto desconexo, diga: "ERRO: O documento analisado não parece ser um Edital válido."
        3. Se a informação da pergunta NÃO estiver escrita no documento, diga: "IRREGULARIDADE/OMISSÃO: Este item não foi localizado no texto do edital." (NÃO assuma que existe só porque está na lei).
        4. Se encontrar, cite o trecho e compare com a Lei/Jurisprudência.
        5. NÃO COLOQUE ASSINATURA.
        
        Contexto Legal (Use apenas para comparar, não para inventar fatos): {context}
        PARECER TÉCNICO:
        """

    elif doc_type == "ETP":
        prompt_template = """
        Você é um Auditor de Planejamento.
        Analise o texto do DOCUMENTO UPLOADED (ETP).
        
        PERGUNTA DA AUDITORIA: {question}
        
        REGRAS CRÍTICAS:
        1. BASEIE-SE ESTRITAMENTE NO TEXTO DO DOCUMENTO.
        2. Se o item (ex: Estimativa de Valor) não estiver escrito explicitamente no documento, diga: "OMISSÃO: O documento não apresenta este tópico obrigatório."
        3. Se o documento for inválido (boleto, imagem), avise o usuário.
        4. NÃO COLOQUE ASSINATURA.
        
        Contexto Legal: {context}
        PARECER SOBRE O ETP:
        """

    else: # TR
        prompt_template = """
        Você é um Auditor Técnico.
        Analise o texto do DOCUMENTO UPLOADED (TR/Projeto Básico).
        
        PERGUNTA DA AUDITORIA: {question}
        
        REGRAS CRÍTICAS:
        1. Busque a evidência APENAS no texto do documento fornecido.
        2. Se não encontrar a definição do objeto ou fiscalização, diga: "OMISSÃO: Tópico não localizado no texto."
        3. Não invente informações que não estão no PDF.
        4. NÃO COLOQUE ASSINATURA.
        
        Contexto Legal: {context}
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

# --- 4. PROCESSAMENTO ---
def process_audit(vectorstore, uploaded_file, doc_type, questions_list):
    reader = PdfReader(uploaded_file)
    doc_text = ""
    for page in reader.pages:
        doc_text += page.extract_text()
    
    # Validação Mínima de Texto
    if len(doc_text) < 50:
        return [("Erro de Leitura", "O arquivo PDF parece ser uma imagem ou está vazio/protegido. Não foi possível ler o texto.")]

    chain = get_specialized_chain(doc_type)
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (titulo_bonito, prompt_tecnico) in enumerate(questions_list):
        status_text.text(f"Analisando: {titulo_bonito}...")
        docs = vectorstore.similarity_search(prompt_tecnico)
        
        # AQUI O SEGREDINHO: Reforçamos no input que o Texto do Documento é a Verdade
        resp = chain.run(input_documents=docs, question=f"Texto do Documento (FONTE DA VERDADE): {doc_text[:6000]}... PERGUNTA DE AUDITORIA: {prompt_tecnico}")
        
        results.append((titulo_bonito, resp))
        progress_bar.progress((i + 1) / len(questions_list))
    
    status_text.text("Concluído!")
    return results

# --- 5. EXIBIÇÃO ---
def display_results(results_list, doc_type):
    if results_list:
        st.markdown(f"### 📋 Resultado da Análise ({doc_type})")
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
        with st.spinner("Inicializando o sistema..."):
            vectorstore, qtd, logs = load_knowledge_base()
        
        if st.session_state.get('user_key') == "GUSTAVO_ADMIN" and qtd > 0:
             with st.expander("🕵️ Logs do Admin"):
                for log in logs: st.write(log)
        
        if vectorstore:
            tab1, tab2, tab3 = st.tabs(["📄 EDITAL", "📘 ETP", "📋 TR / P. BÁSICO"])
            
            # --- ABA 1: EDITAL ---
            with tab1:
                file_edital = st.file_uploader("Selecione o arquivo PDF do Edital", type="pdf", key="u1")
                if file_edital and st.button("AUDITAR ARQUIVO (1 Crédito)", key="b1"):
                    questions = [
                        ("1. Análise de Modalidade e Critério", "Verifique a MODALIDADE e o CRITÉRIO DE JULGAMENTO no texto. Estão adequados ao objeto? (Art. 28 e 33)"),
                        ("2. Análise de Habilitação", "Analise os REQUISITOS DE HABILITAÇÃO (Jurídica, Fiscal, Técnica, Econômica) descritos no texto. Há excessos ou restrições?"),
                        ("3. Prazos e Publicidade", "Busque no texto os PRAZOS DE PUBLICAÇÃO e de IMPUGNAÇÃO. Eles respeitam os dias úteis exigidos pela Lei 14.133?")
                    ]
                    st.session_state['result_edital'] = process_audit(vectorstore, file_edital, "EDITAL", questions)
                
                if st.session_state['result_edital']:
                    display_results(st.session_state['result_edital'], "EDITAL")

            # --- ABA 2: ETP ---
            with tab2:
                file_etp = st.file_uploader("Selecione o arquivo PDF do ETP", type="pdf", key="u2")
                if file_etp and st.button("AUDITAR ARQUIVO (1 Crédito)", key="b2"):
                    questions = [
                        ("1. Análise da Necessidade", "O texto descreve a NECESSIDADE da contratação de forma clara? (Inciso I)"),
                        ("2. Levantamento de Mercado", "O texto comprova que houve LEVANTAMENTO DE MERCADO e análise de alternativas? (Inciso III)"),
                        ("3. Estimativa e Orçamento", "O texto apresenta a ESTIMATIVA DO VALOR e adequação orçamentária? (Inciso VI e VII)"),
                        ("4. Justificativa da Solução", "A ESCOLHA DA SOLUÇÃO foi justificada no texto? (Inciso VIII)")
                    ]
                    st.session_state['result_etp'] = process_audit(vectorstore, file_etp, "ETP", questions)
                
                if st.session_state['result_etp']:
                    display_results(st.session_state['result_etp'], "ETP")

            # --- ABA 3: TR ---
            with tab3:
                file_tr = st.file_uploader("Selecione o arquivo PDF do TR", type="pdf", key="u3")
                if file_tr and st.button("AUDITAR ARQUIVO (1 Crédito)", key="b3"):
                    questions = [
                        ("1. Definição do Objeto", "A definição do OBJETO no texto é precisa e suficiente? Há vedação de marca?"),
                        ("2. Modelo de Execução", "O texto detalha o MODELO DE EXECUÇÃO do objeto?"),
                        ("3. Medição e Pagamento", "Os CRITÉRIOS DE MEDIÇÃO E PAGAMENTO estão escritos no texto?"),
                        ("4. Fiscalização", "Há cláusula de FISCALIZAÇÃO e critérios de recebimento no texto?")
                    ]
                    st.session_state['result_tr'] = process_audit(vectorstore, file_tr, "TR", questions)
                
                if st.session_state['result_tr']:
                    display_results(st.session_state['result_tr'], "TR")

if __name__ == "__main__":
    main()
