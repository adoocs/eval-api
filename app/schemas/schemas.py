from pydantic import BaseModel
from typing import Optional, List

class LoanApplication(BaseModel):
    ID: str
    edad: int
    genero: str
    estado_civil: str
    tipo_vivienda: str
    integrantes: int
    zona_geografica: str
    tipo_zona: str
    monto_solicitado: float
    plazo: int
    periodo: str
    ingreso_adicional: float
    ingreso_terceros: float
    sector_economico: str
    ingreso_mensual_bruto: float
    costo_inversion: float
    ingreso_mensual_neto: float
    gasto_basico: float
    gastos_operativos: float
    gastos_financieros: float
    deuda_total_financiera: float
    puntaje_experian: int

class PredictionResult(BaseModel):
    ID: str
    prediccion: int
    probabilidad_aprobado: float
    probabilidad_rechazado: float
    confianza: float
    mensaje: str

class BatchPredictionRequest(BaseModel):
    applications: List[LoanApplication]

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResult]
    total_processed: int
    errors: List[str]

class TrainingResponse(BaseModel):
    message: str
    accuracy: float
    model_path: str