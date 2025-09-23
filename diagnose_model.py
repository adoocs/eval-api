import joblib
import json
import pandas as pd
from app.models.model_trainer import ModelTrainer

def diagnose_model():
    """Diagnosticar problemas con el modelo"""
    try:
        # Cargar metadata del modelo
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        feature_columns = metadata.get('feature_columns', [])
        
        print("🔍 DIAGNÓSTICO DEL MODELO")
        print("=" * 50)
        print(f"📊 Número de features: {len(feature_columns)}")
        print(f"📋 Primeros 10 features: {feature_columns[:10]}")
        
        # Verificar si hay features de ID
        id_features = [col for col in feature_columns if 'ID_' in col or 'SOL-' in col]
        print(f"🚨 Features de ID detectados: {len(id_features)}")
        if id_features:
            print(f"   Ejemplos: {id_features[:5]}")
        
        # Verificar features problemáticos
        problematic_features = [col for col in feature_columns if len(col) > 50]
        print(f"⚠️  Features muy largos: {len(problematic_features)}")
        
    except Exception as e:
        print(f"❌ Error en diagnóstico: {e}")

def retrain_model_fixed():
    """Re-entrenar el modelo con la corrección"""
    print("🔄 Re-entrenando modelo con corrección...")
    
    trainer = ModelTrainer()
    accuracy = trainer.train("../data/dataset_micro.csv")
    trainer.save_model()
    
    print(f"✅ Modelo re-entrenado. Exactitud: {accuracy:.4f}")

if __name__ == "__main__":
    diagnose_model()
    print("\n" + "="*50)
    retrain_model_fixed()