from datetime import datetime
from fastapi import APIRouter, HTTPException
from app.services.unified_predictor import UnifiedPredictor
from app.schemas.prediction_schemas import PredictionResponse
from app.schemas.schemas import LoanApplication

router = APIRouter(prefix="/api/v1", tags=["Predicción Unificada"])
predictor = UnifiedPredictor()

@router.post("/predict-completo", response_model=PredictionResponse)
async def predict_completo(application: LoanApplication):
    """
    Predicción completa con ML + Análisis de Riesgo + Métricas
    Ideal para almacenar en base de datos para reportes
    """
    try:
        datos = application.dict()
        resultado = predictor.predecir_completo(datos)
        
        return PredictionResponse(**resultado)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {str(e)}")

@router.post("/predict-batch-completo")
async def predict_batch_completo(applications: list[LoanApplication]):
    """
    Predicción por lote con todas las métricas
    """
    try:
        resultados = []
        for app in applications:
            datos = app.dict()
            resultado = predictor.predecir_completo(datos)
            resultados.append(resultado)
        
        return {
            "total_procesado": len(resultados),
            "resultados": resultados,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))