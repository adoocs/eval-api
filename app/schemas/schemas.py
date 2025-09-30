from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class LoanApplication(BaseModel):
    ID: str
    edad: int = Field(..., ge=18, le=80, description="Edad entre 18 y 80 años")
    genero: str = Field(..., pattern='^(F|M)$', description="Género: F o M")
    estado_civil: str = Field(..., description="Estado civil: soltero, casado, viudo, divorciado, conviviente")
    tipo_vivienda: str = Field(..., description="Tipo de vivienda: Propia, Familiar, Alquilada")
    integrantes: int = Field(..., ge=1, le=10, description="Número de integrantes del hogar")
    zona_geografica: str = Field(..., description="Zona geográfica: Costa, Sierra, Selva")
    tipo_zona: str = Field(..., description="Tipo de zona: Urbano, Periurbano, Rural")
    monto_solicitado: float = Field(..., ge=100, le=50000, description="Monto solicitado del préstamo")
    plazo: int = Field(..., ge=1, le=120, description="Plazo del préstamo")
    periodo: str = Field(..., description="Periodo de pago: Diario, Semanal, Quincenal, Mensual")
    sector_economico: str = Field(..., description="Sector económico: Comercio, Produccion, Servicios, Transporte")
    ingreso_mensual_bruto: Optional[float] = Field(None, ge=0, description="Ingreso mensual bruto")
    costo_inversion: Optional[float] = Field(None, ge=0, le=100, description="Costo de inversión como porcentaje")
    ingreso_mensual_neto: float = Field(..., ge=0, description="Ingreso mensual neto")
    ingreso_adicional: float = Field(..., ge=0, description="Ingresos adicionales")
    ingreso_terceros: float = Field(..., ge=0, description="Ingresos de terceros")
    ingreso_total: Optional[float] = Field(None, ge=0, description="Ingreso total calculado")
    gasto_basico: float = Field(..., ge=0, description="Gastos básicos")
    gastos_operativos: float = Field(..., ge=0, description="Gastos operativos")
    gastos_financieros: float = Field(..., ge=0, description="Gastos financieros")
    gasto_total: Optional[float] = Field(None, ge=0, description="Gasto total calculado")
    puntaje_experian: int = Field(..., ge=0, le=999, description="Puntaje Experian (0-999)")
    
    # Nuevos campos para métricas de riesgo
    # capacidad_pago_porcentaje: Optional[float] = Field(None, ge=-100, le=100, description="Capacidad de pago en porcentaje")
    #indice_endeudamiento: Optional[float] = Field(None, ge=0, le=500, description="Índice de endeudamiento")
    #ajuste_historial: Optional[float] = Field(None, ge=0, le=40, description="Ajuste por historial crediticio")
    #riesgo_crediticio: Optional[float] = Field(None, ge=0, le=100, description="Riesgo crediticio total")

class RiskMetrics(BaseModel):
    capacidad_pago_porcentaje: float = Field(..., ge=-100, le=100)
    indice_endeudamiento: float = Field(..., ge=0, le=500)
    ajuste_historial: float = Field(..., ge=0, le=40)
    riesgo_crediticio: float = Field(..., ge=0, le=100)

class PredictionResult(BaseModel):
    ID: str
    prediccion: int = Field(..., ge=0, le=1, description="0=Rechazado, 1=Aprobado")
    probabilidad_aprobado: float = Field(..., ge=0, le=1)
    probabilidad_rechazado: float = Field(..., ge=0, le=1)
    confianza: float = Field(..., ge=0, le=1)
    mensaje: str
    razon: str = Field(..., description="Razón de la decisión")
    metricas_riesgo: RiskMetrics
    modelo_metricas: Optional[Dict[str, Any]] = Field(None, description="Métricas del modelo entrenado")
    tiempo_procesamiento_segundos: Optional[float] = Field(None, description="Tiempo de procesamiento en segundos")
    tiempo_procesamiento_ms: Optional[float] = Field(None, description="Tiempo de procesamiento en milisegundos")

class BatchPredictionRequest(BaseModel):
    applications: List[LoanApplication]

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResult]
    total_processed: int = Field(..., ge=0)
    errors: List[str]
    summary: Optional[Dict[str, Any]] = Field(None, description="Resumen de las predicciones")

class TrainingResponse(BaseModel):
    message: str
    accuracy: float = Field(..., ge=0, le=1)
    precision: float = Field(..., ge=0, le=1)
    recall: float = Field(..., ge=0, le=1)
    f1_score: float = Field(..., ge=0, le=1)
    model_path: str
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Importancia de las características")
    training_size: int = Field(..., ge=0)

class ModelInfoResponse(BaseModel):
    cargado: bool
    caracteristicas: int = Field(..., ge=0)
    metricas: Dict[str, Any]
    umbral_riesgo: float = Field(..., ge=0, le=1)
    fecha_entrenamiento: Optional[str] = Field(None, description="Fecha del último entrenamiento")

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
    code: Optional[int] = None

# Modelos para análisis de métricas de tu tesis
class RiskAnalysis(BaseModel):
    porcentaje_precision: float = Field(..., ge=0, le=100, description="Porcentaje de precisión")
    porcentaje_riesgo: float = Field(..., ge=0, le=100, description="Porcentaje de riesgo promedio")
    distribucion_riesgo: Dict[str, float] = Field(..., description="Distribución por niveles de riesgo")
    tiempo_procesamiento: Optional[float] = Field(None, description="Tiempo promedio de procesamiento")

class ThesisMetricsResponse(BaseModel):
    precision_analisis_crediticio: float = Field(..., ge=0, le=100)
    riesgo_promedio_aprobacion: float = Field(..., ge=0, le=100)
    tiempo_evaluacion_promedio: Optional[float] = Field(None)
    satisfaccion_asesores: Optional[float] = Field(None, ge=1, le=5)
    analisis_riesgo: RiskAnalysis
    comparativo_modelo_vs_tradicional: Optional[Dict[str, Any]] = Field(None)