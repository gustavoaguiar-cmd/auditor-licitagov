import streamlit as st
import os
from pypdf import PdfReader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Carrega variáveis de ambiente (Local ou Railway)
load_dotenv()

# --- CONFIGURAÇÃO DE SEGURANÇA (LOGIN) ---
# Em produção, idealmente usaríamos um banco de dados.
CLIENTES_AUTORIZADOS = {
    "admin": "admin123",        # Acesso Mestre
    "cliente": "solar2025",     # Exemplo de Cliente
    "teste": "123456"           # Acesso Teste
}

def check_login():
    """Gerencia o acesso ao sistema via barra lateral"""
    st.sidebar.title("🔐 Área do Cliente")
    
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        usuario = st.sidebar.text_input("Usuário")
        senha = st.sidebar.text_input("Senha", type="password")
        
        if st.sidebar.button("Entrar"):
            if usuario in CLIENTES_AUTORIZADOS and CLIENTES_AUTORIZADOS[usuario] == senha:
                st.session_state["logged_in"] = True
                st.session_state["usuario_atual"] = usuario
                st.rerun()
            else:
                st.sidebar.error("Usuário ou senha incorretos.")
        return False
    else:
        st.sidebar.success(f"Auditor Logado: {st.session_state['usuario_atual']}")
        if st.sidebar.button("Sair"):
            st.session_state["logged_in"] = False
            st.rerun()
        return True

# --- FUNÇÕES DE INTELIGÊNCIA (RAG & LEITURA) ---

@st.cache_resource
def load_knowledge_base():
    """
    Lê PDFs recursivamente em data/legislacao e TODAS as suas subpastas.
    Cria a memória vetorial (FAISS) para o Auditor consultar.
    """
    docs = []
    folder_path = "data/legislacao"
    
    # Verifica se a pasta existe
    if not os.path.exists(folder_path):
        return None

    # Caminha por todas as subpastas (os.walk)
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(root, filename)
                try:
                    reader = PdfReader(file_path)
                    text = ""
                    for page in reader.pages:
                        if page.extract_text():
                            text += page.extract_text()
                    
                    if text: # Só adiciona se conseguiu extrair texto
                        # Adiciona metadados com o nome do arquivo para citação
                        docs.append(Document(page_content=text, metadata={"source": filename}))
                except Exception as e:
                    print(f"Erro ao ler arquivo {filename}: {e}")
                    pass
    
    if not docs:
        return None

    # Quebra os textos em pedaços menores para a IA processar
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Cria o Banco Vetorial
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

def get_pdf_text(pdf_docs):
    """Extrai texto do PDF enviado pelo usuário"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def get_audit_prompt(doc_type):
    """
    Retorna o Prompt de Auditoria Específico para cada tipo de documento,
    garantindo que a IA foque nas regras corretas da Lei 14.133/21.
    """
    
    # Cabeçalho comum (Define a Persona e injeta a Jurisprudência)
    header = """
    Você é um Auditor de Controle Externo Sênior (perfil rigoroso TCE/ES e TCU).
    Sua missão é cruzar o documento analisado com a Lei 14.133/2021 e a JURISPRUDÊNCIA fornecida.
    Não seja superficial. Aponte riscos, erros e omissões com base legal.

    CONTEXTO JURÍDICO (Use estas fontes para fundamentar sua análise):
    {context}

    DOCUMENTO EM ANÁLISE ({doc_type}):
    {text}
    """

    if doc_type == "Edital de Licitação":
        return header + """
        ---
        DIRETRIZES DE ANÁLISE COMPLETA (EDITAL):
        
        1. ASPECTOS LEGAIS E ESTRUTURAIS:
           - Fundamentação: Cita corretamente a Lei 14.133/21?
           - Objeto: É claro e preciso? Há indícios de direcionamento?
           - Minuta do Contrato e Matriz de Risco: CONSTAM? A ausência de Matriz de Risco em obras/serviços complexos é FALHA GRAVE.
        
        2. HABILITAÇÃO (O Ponto Crítico - Cruzar com Jurisprudência):
           - Qualificação Técnica: 
             * Atestados: A exigência (ex: quantitativos mínimos) ultrapassa 50% do objeto? (Súmula TCU 263).
             * Certificações (ISO, CMVP, PBQP-H): São eliminatórias? Se sim, marque como RISCO se não houver justificativa robusta técnica.
           - Qualificação Econômica:
             * Capital Social/PL: Exige mais de 10% do valor estimado? (Ilegal - Art. 69).
             * Índices (LG, SG, LC): São usuais (>1.0)?

        3. ORÇAMENTO E JULGAMENTO:
           - Critério de Julgamento: Está definido (Menor Preço, Maior Desconto)?
           - Orçamento: Menciona se é sigiloso ou aberto? O BDI está adequado?
           - Exigência de Garantia de Proposta (1%) e Contratual (5% a 10%): Estão previstas?

        4. MODOS DE DISPUTA E PRAZOS:
           - O modo de disputa (Aberto/Fechado) está claro?
           - Os prazos de publicidade respeitam a Lei 14.133 (min. 10 dias úteis pregão comum, etc)?

        FORMATO DO RELATÓRIO FINAL:
        ### 🎯 1. Resumo Executivo
        (Parecer geral sobre a legalidade e os principais riscos).

        ### 🔍 2. Auditoria Detalhada (Ponto a Ponto)
        * **Item Analisado:** (Ex: 10.4 - Qualificação Técnica)
        * **Análise:** (Sua análise técnica).
        * **Cruzamento Jurídico:** (Cite o Acórdão/Manual do banco de dados que embasa sua opinião).
        * **Veredito:** ✅ REGULAR / ⚠️ RISCO JUSTIFICADO / 🚨 IRREGULAR / ❌ AUSENTE

        ### ⚖️ 3. Análise Econômica e Orçamentária
        
        ### 📝 4. Recomendações Corretivas
        (Liste o que o gestor deve alterar ou justificar para evitar impugnação).
        """

    elif doc_type == "Estudo Técnico Preliminar (ETP)":
        return header + """
        ---
        DIRETRIZES DE ANÁLISE (ETP - Art. 18, §1º):
        
        1. NECESSIDADE E SOLUÇÃO:
           - O problema a ser resolvido está bem descrito?
           - Levantamento de Mercado: Houve comparação de diferentes soluções? Se indicou apenas uma solução sem comparar, aponte RISCO DE DIRECIONAMENTO.
        
        2. PARCELAMENTO (Súmula 247 TCU):
           - O ETP justifica técnica e economicamente o parcelamento ou não do objeto? A ausência dessa justificativa é ERRO.
        
        3. VIABILIDADE E PREVISÃO:
           - Há estimativa do valor da contratação?
           - Demonstra alinhamento com o PCA (Plano de Contratações Anual)?
        
        GERE O RELATÓRIO FOCANDO NOS INCISOS DO ART. 18 QUE FALTAM OU ESTÃO GENÉRICOS.
        """

    elif doc_type == "Termo de Referência (TR)":
        return header + """
        ---
        DIRETRIZES DE ANÁLISE (TR - Art. 6º, XXIII):
        
        1. DEFINIÇÃO DO OBJETO:
           - A descrição é precisa e suficiente para o licitante precificar?
           - Vistoria Técnica: É obrigatória? Se sim, há justificativa? (O TCU recomenda que seja facultativa).
        
        2. MODELO DE EXECUÇÃO E GESTÃO:
           - Define como o serviço será executado?
           - Define o fiscal do contrato e suas atribuições?
        
        3. PAGAMENTO E CRITÉRIOS:
           - Critérios de medição estão claros? (Pagamento por resultado vs Pagamento fixo).
           - Exige adequação orçamentária?
        
        GERE O RELATÓRIO APONTANDO CLÁUSULAS OBSCURAS OU OMISSAS NO TR.
        """

    else: # Projeto Básico
        return header + """
        ---
        DIRETRIZES DE ANÁLISE (PROJETO BÁSICO - Obras/Serviços Engenharia):
        
        1. ELEMENTOS TÉCNICOS (Lei 14.133, Art. 6º, XXV):
           - Levantamentos topográficos e sondagens: Foram realizados ou citados? (Essencial para evitar aditivos).
           - Memorial Descritivo: É detalhado?
        
        2. ORÇAMENTO:
           - Orçamento detalhado em planilha (custos unitários)?
           - Cronograma físico-financeiro existe?
        
        3. BDI E ENCARGOS:
           - O BDI está detalhado?
        
        GERE O RELATÓRIO FOCANDO NA PRECISÃO DO PROJETO PARA EVITAR OBRAS PARADAS.
        """

# --- INTERFACE PRINCIPAL DO SISTEMA ---

st.set_page_config(page_title="Lici Auditor v12 - Expert", page_icon="⚖️", layout="wide")

# CSS para deixar com cara de SaaS Profissional
st.markdown("""
<style>
    .stApp {background-color: #f8f9fa;}
    h1 {color: #0f2c4a;}
    h2, h3 {color: #1c4b75;}
    .stAlert {border-left: 5px solid #ff4b4b;}
    div[data-testid="stSidebar"] {background-color: #eef2f5;}
</style>
""", unsafe_allow_html=True)

# 1. VERIFICA LOGIN (Bloqueia execução se não logar)
if not check_login():
    st.stop()

# 2. CARREGA CONFIGURAÇÕES E MEMÓRIA
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("ERRO CRÍTICO: API Key não configurada no servidor (Railway).")
    st.stop()

# Barra Lateral com Status da Base de Conhecimento
with st.sidebar:
    st.markdown("---")
    st.write("📚 **Base de Conhecimento (RAG):**")
    
    with st.spinner("Indexando Legislação e Jurisprudência..."):
        vectorstore = load_knowledge_base()
    
    if vectorstore:
        st.success("✅ Biblioteca Jurídica Ativa")
        st.caption("Fontes: Manuais TCU, Acórdãos, Leis (lendo subpastas).")
    else:
        st.warning("⚠️ Nenhuma base encontrada em data/legislacao")
        st.info("O sistema usará apenas o conhecimento geral da IA.")

# Corpo Principal
st.title("Lici Auditor v12 🏛️ (Expert Mode)")
st.markdown("### Auditoria Jurídica Inteligente - Lei 14.133/21")

# Seleção do Tipo de Documento
col1, col2 = st.columns([1, 2])
with col1:
    doc_type = st.selectbox(
        "Tipo de Documento:",
        ["Edital de Licitação", "Estudo Técnico Preliminar (ETP)", "Termo de Referência (TR)", "Projeto Básico"]
    )

# Upload
uploaded_file = st.file_uploader("Faça upload do documento (PDF)", type="pdf")

if uploaded_file and st.button("🔍 Iniciar Auditoria Profunda"):
    with st.spinner(f"O Auditor está analisando o {doc_type} e cruzando com a Jurisprudência..."):
        try:
            # 1. Extrair Texto do PDF
            raw_text = get_pdf_text([uploaded_file])
            
            if len(raw_text) < 100:
                st.error("O arquivo parece vazio ou é uma imagem digitalizada. Preciso de PDF com texto selecionável.")
            else:
                # 2. Busca Inteligente (RAG) - Trazendo o contexto jurídico
                contexto_juridico = ""
                if vectorstore:
                    # Busca os 6 trechos mais relevantes no banco de dados
                    docs_rel = vectorstore.similarity_search(raw_text[:4000], k=6) 
                    for doc in docs_rel:
                        # Monta o texto de referência citando a fonte
                        fonte = doc.metadata.get('source', 'Desconhecida')
                        contexto_juridico += f"\n[FONTE: {fonte}]\n...{doc.page_content[:600]}...\n"
                else:
                    contexto_juridico = "Base de conhecimento local não disponível. Usando conhecimento geral da Lei 14.133."

                # 3. Configura a IA (GPT-4)
                # temperature=0.1 para ser criativo na análise mas rigoroso nos fatos
                llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.1, openai_api_key=api_key)
                
                # 4. Seleciona o Prompt Correto
                prompt_text = get_audit_prompt(doc_type)
                
                # Prepara o template passando doc_type também
                prompt = PromptTemplate(template=prompt_text, input_variables=["context", "text", "doc_type"])
                
                # Monta o prompt final (Limitando caracteres para não estourar tokens)
                final_prompt = prompt.format(
                    context=contexto_juridico, 
                    text=raw_text[:70000], # Aumentei o limite de leitura
                    doc_type=doc_type
                )
                
                # 5. Executa
                response = llm.invoke(final_prompt)
                
                # 6. Exibe Resultado
                st.success("Auditoria Concluída!")
                st.markdown(response.content)
                
                # Botão de Download
                st.download_button(
                    label="📥 Baixar Relatório Completo",
                    data=response.content,
                    file_name=f"Auditoria_{doc_type.split()[0]}.md",
                    mime="text/markdown"
                )
                
        except Exception as e:
            st.error(f"Ocorreu um erro durante a análise: {e}")
