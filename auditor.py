import streamlit as st
import os
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Configuração da Página
st.set_page_config(page_title="Lici Auditor v10", page_icon="⚖️", layout="wide")

# CSS para visual profissional (Esconde menus de dev e melhora a UI)
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {background-color: #f8f9fa;}
    .css-1d391kg {padding-top: 1rem;}
    .stAlert {font-weight: bold;}
    h1 {color: #0f2c4a;}
    h2 {color: #1c4b75;}
</style>
""", unsafe_allow_html=True)

# Barra Lateral
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/10325/10325149.png", width=100)
    st.title("Lici Auditor ⚖️")
    st.info("Versão 10.0 - Lei 14.133/21")
    st.markdown("---")
    api_key = st.text_input("Insira sua API Key OpenAI:", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

# Funções Auxiliares
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_audit_prompt(doc_type):
    # PROMPTS ESPECIALIZADOS BASEADOS NA LEI 14.133
    
    if doc_type == "Edital de Licitação":
        return """
        Você é um Auditor Especialista em Licitações Públicas no Brasil (Lei 14.133/2021).
        Analise o texto do EDITAL abaixo com rigor extremo.
        
        Sua análise deve ser dividida nas seguintes seções obrigatórias:

        1. ASPECTOS LEGAIS E ESTRUTURAIS
        - Verifique se cita a Lei 14.133/2021.
        - Objeto: Está claro e sem direcionamento de marca?
        - Critério de Julgamento: Está definido?
        - Minuta do Contrato e Matriz de Risco: CONSTAM? Se não, aponte como FALHA GRAVE.
        - Orçamento/Reajuste: Se não encontrar aqui, diga "Não encontrado no Edital - Verificar TR".

        2. HABILITAÇÃO E PARTICIPAÇÃO (Foco em Restrições)
        - Verifique Habilitação Jurídica, Fiscal, Social, Trabalhista.
        - ALERTA VERMELHO: Procure por exigências restritivas (ex: comprovação de regularidade APENAS para assinatura do contrato vs habilitação). Cite jurisprudência se houver restrição indevida.
        - Qualificação Econômica: Índices são proporcionais?
        - Qualificação Técnica: Atestados são compatíveis?

        3. REQUISITOS ESSENCIAIS
        - Amostras: Se exigidas, há regra clara?
        - Modo de Disputa: Aberto/Fechado definido?
        - Prazos: Respeitam a Lei 14.133?
        - ME/EPP: Prevê tratamento diferenciado?

        TEXTO DO DOCUMENTO:
        {text}

        SAÍDA ESPERADA:
        Para cada item, diga "CONFORME" ou "NÃO CONFORME/AUSENTE".
        Se encontrar cláusula restritiva ou ilegal, inicie a linha com "🚨 ALERTA VERMELHO:".
        Cite o artigo da lei ou jurisprudência (TCU) aplicável em cada análise negativa.
        Ao final, faça um "RELATÓRIO DE PENDÊNCIAS" resumindo o que falta.
        """

    elif doc_type == "Estudo Técnico Preliminar (ETP)":
        return """
        Você é um Auditor da Lei 14.133/21. Analise este ETP com base estrita no Art. 18, §1º.
        Verifique a presença e qualidade de CADA um dos seguintes incisos:

        I - Descrição da necessidade (Problema a ser resolvido).
        II - Previsão no PCA (Plano de Contratações Anual).
        III - Requisitos da contratação.
        IV - Estimativas de quantidades (com memória de cálculo).
        V - Levantamento de mercado e justificativa da solução.
        VI - Estimativa do valor (com preços unitários).
        VII - Descrição da solução como um todo.
        VIII - Justificativa de parcelamento.
        IX - Resultados pretendidos (economicidade/eficiência).
        X - Providências prévias (inclusive capacitação).
        XI - Contratações correlatas.
        XII - Impactos ambientais e medidas mitigadoras.
        XIII - Posicionamento conclusivo sobre adequação.

        TEXTO DO DOCUMENTO:
        {text}

        SAÍDA ESPERADA:
        Liste inciso por inciso. Se estiver ausente ou genérico, marque com "🚨 ERRO".
        Cite o Art. 18 da Lei 14.133 em itens faltantes.
        """

    elif doc_type == "Termo de Referência (TR)":
        return """
        Você é um Auditor da Lei 14.133/21. Analise este Termo de Referência (TR) com base no Art. 6º, XXIII.
        Verifique obrigatoriamente:

        a) Definição do objeto, quantitativos e prazo.
        b) Fundamentação (referência ao ETP).
        c) Descrição da solução (ciclo de vida).
        d) Requisitos da contratação.
        e) Modelo de execução.
        f) Modelo de gestão (fiscalização).
        g) Critérios de medição e pagamento.
        h) Forma de seleção do fornecedor.
        i) Estimativas de valor e adequação orçamentária (Se não constava no edital, É OBRIGATÓRIO AQUI).

        TEXTO DO DOCUMENTO:
        {text}

        SAÍDA ESPERADA:
        Analise item a item. Se faltar a adequação orçamentária ou reajuste aqui (e não estava no edital), gere um ALERTA CRÍTICO.
        """
    
    else: # Projeto Básico
        return """
        Analise este Projeto Básico com base no Art. 6º, XXV da Lei 14.133/21.
        Verifique: Levantamentos topográficos, soluções técnicas, tipos de serviços, métodos construtivos, orçamento detalhado (custo global).
        
        TEXTO DO DOCUMENTO:
        {text}
        """

# Interface Principal
st.title("Lici Auditor v10 🏛️")
st.markdown("### Auditoria Jurídica Inteligente - Lei 14.133/21")

# Seleção do Tipo de Documento
doc_type = st.selectbox(
    "Qual documento você vai auditar?",
    ["Edital de Licitação", "Estudo Técnico Preliminar (ETP)", "Termo de Referência (TR)", "Projeto Básico"]
)

# Upload de Arquivo
uploaded_file = st.file_uploader("Faça upload do documento (PDF)", type="pdf")

if uploaded_file and st.button("🔍 Iniciar Auditoria Blindada"):
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Por favor, insira a API Key na barra lateral.")
    else:
        with st.spinner(f"Lendo documento e cruzando com a Lei 14.133 e Jurisprudência..."):
            # 1. Extrair Texto
            raw_text = get_pdf_text([uploaded_file])
            
            # 2. Preparar a IA
            llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0) # Usando GPT-4 Turbo para maior precisão jurídica
            
            # 3. Selecionar Prompt
            audit_prompt = get_audit_prompt(doc_type)
            prompt = PromptTemplate(template=audit_prompt, input_variables=["text"])
            
            # 4. Executar Análise
            try:
                final_prompt = prompt.format(text=raw_text[:80000]) # Limite seguro de caracteres
                response = llm.invoke(final_prompt)
                
                # 5. Exibir Resultado
                st.success("Auditoria Concluída!")
                st.markdown("### 📋 Relatório de Análise")
                st.markdown(response.content)
                
                # Botão para baixar relatório
                st.download_button(
                    label="📥 Baixar Relatório",
                    data=response.content,
                    file_name=f"Auditoria_{doc_type}.md",
                    mime="text/markdown"
                )
                
            except Exception as e:
                st.error(f"Erro durante a análise: {e}")
