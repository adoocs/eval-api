from pydantic import BaseModel
from typing import Dict, Any

class PredictionResponse(BaseModel):
    # Identificación
    ID: str
    timestamp: str
    
    # Predicción ML
    prediccion_ml: int
    probabilidad_aprobado: float
    probabilidad_rechazado: float
    confianza_ml: float
    mensaje: str
    
    # Análisis de Riesgo (NUEVO)
    capacidad_pago: float
    nivel_riesgo: str
    recomendacion_riesgo: str
    criterio_empresa: str
    
    # Métricas de Evaluación (NUEVO)
    metricas_riesgo: Dict[str, float]
    variables_clave: Dict[str, Any]
    
    # Decision Final
    decision_final: str
    motivo: str