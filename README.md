# Crear entorno virtual
python -m venv venv

# Activar entorno virtual (Windows)
venv\Scripts\activate
# source venv/bin/activate  # Linux/Mac

# Instalar desde requirements.txt
pip install -r requirements.txt

# Opción 1: Usar el script de entrenamiento
python train_model.py

# Levantar el servidor de desarrollo
python run_api.py

# O directamente con uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000