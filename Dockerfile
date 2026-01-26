# Usa uma imagem leve do Python
FROM python:3.9-slim

# Define o diretório de trabalho
WORKDIR /app

# Instala ferramentas básicas do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copia os arquivos do projeto para o servidor
COPY . .

# Instala as bibliotecas do requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta padrão do Streamlit
EXPOSE 8501

# Comando de verificação de saúde (Healthcheck)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Comando para iniciar o Auditor
ENTRYPOINT ["streamlit", "run", "auditor.py", "--server.port=8501", "--server.address=0.0.0.0"]
