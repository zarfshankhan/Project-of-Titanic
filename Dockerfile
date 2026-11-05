# Official Python lightweight image use karein
FROM python:3.9-slim

# Working directory set karein container ke andar
WORKDIR /app

# Pehle requirements copy karein taaki cache use ho sake
COPY requirements.txt .

# Dependencies install karein
RUN pip install --no-cache-dir -r requirements.txt

# Baaki saari files copy karein
COPY . .

# Streamlit ka default port expose karein
EXPOSE 8501

# Streamlit app run karein
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]