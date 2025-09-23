from fastapi import APIRouter, HTTPException
from app.schemas.schemas import (
    LoanApplication, PredictionResult, BatchPredictionRequest,
    BatchPredictionResponse, TrainingResponse
)
from app.models.predictor import LoanPredictor
from app.models.model_trainer import ModelTrainer
import os

router = APIRouter()

# Instancia global del predictor
predictor = LoanPredictor()

@router.get("/")
async def root():
    return {"message": "API de Predicción de Aprobación de Préstamos"}

@router.get("/health")
async def health_check():
    """Verificar estado del modelo"""
    return {
        "status": "healthy" if predictor.is_loaded else "model_not_loaded",
        "model_loaded": predictor.is_loaded
    }

@router.post("/predict", response_model=PredictionResult)
async def predict_single(application: LoanApplication):
    """Predecir para una sola solicitud - MEJORADO"""
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Modelo no cargado. Entrene el modelo primero.")
    
    try:
        # Convertir Pydantic model a dict
        app_data = application.dict()
        
        # Validar que no se envíe el ID como feature
        if 'ID' in app_data:
            # El ID se usa solo para identificación, no como feature
            pass  # Será manejado por preprocess_input
        
        result = predictor.predict_single(app_data)
        
        return PredictionResult(
            ID=application.ID,
            **result
        )
    except Exception as e:
        error_detail = f"Error en predicción: {str(e)}"
        print(f"❌ {error_detail}")  # Log para debugging
        raise HTTPException(status_code=400, detail=error_detail)

@router.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_request: BatchPredictionRequest):
    """Predecir para múltiples solicitudes"""
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Modelo no cargado. Entrene el modelo primero.")
    
    try:
        applications = [app.dict() for app in batch_request.applications]
        results, errors = predictor.predict_batch(applications)
        
        return BatchPredictionResponse(
            results=results,
            total_processed=len(results),
            errors=errors
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción batch: {str(e)}")

@router.post("/train", response_model=TrainingResponse)
async def train_model():
    """Entrenar el modelo con los datos actuales"""
    try:
        csv_path = "../data/dataset_micro.csv"
        
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=404, detail="Dataset no encontrado")
        
        trainer = ModelTrainer()
        accuracy = trainer.train(csv_path)
        model_path = trainer.save_model()
        
        # Recargar el predictor con el nuevo modelo
        global predictor
        predictor = LoanPredictor()
        
        return TrainingResponse(
            message="Modelo entrenado y guardado exitosamente",
            accuracy=accuracy,
            model_path=model_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error entrenando modelo: {str(e)}")