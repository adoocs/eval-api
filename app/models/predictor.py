import pandas as pd
import joblib
import json
import os
from typing import Dict, Any, List

class LoanPredictor:
    def __init__(self, model_path="models/random_forest_model.pkl", 
                 metadata_path="models/model_metadata.json"):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.feature_columns = None
        self.is_loaded = False
        self.load_model()
    
    def load_model(self):
        """Cargar modelo y metadata"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modelo no encontrado en: {self.model_path}")
            
            self.model = joblib.load(self.model_path)
            
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            self.feature_columns = metadata['feature_columns']
            self.is_loaded = True
            print("Modelo cargado exitosamente")
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            self.is_loaded = False
    
    def preprocess_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocesar los datos de entrada CORREGIDO"""
        # Crear copia para no modificar el original
        data_copy = input_data.copy()

        # EXTRAER y REMOVER el ID antes del procesamiento
        data_copy.pop('ID', 'N/A')

        # También remover 'aprobado' si existe (para predicción)
        data_copy.pop('aprobado', None)

        # Crear DataFrame SOLO con los features
        df = pd.DataFrame([data_copy])

        # Aplicar One-Hot Encoding
        df_encoded = pd.get_dummies(df)

        # Crear DataFrame base con todas las columnas esperadas
        base_data = {col: 0 for col in self.feature_columns}
        final_df = pd.DataFrame([base_data])

        # Actualizar solo las columnas que existen en los datos de entrada
        for col in df_encoded.columns:
            if col in final_df.columns:
                final_df[col] = df_encoded[col].iloc[0]

        # Verificar que tenemos todas las columnas necesarias
        missing_columns = set(self.feature_columns) - set(final_df.columns)
        if missing_columns:
            print(f"⚠️  Columnas faltantes: {missing_columns}")

        return final_df
    
    def predict_single(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Realizar predicción para una sola solicitud"""
        if not self.is_loaded:
            raise ValueError("Modelo no cargado. Entrene el modelo primero.")
        
        try:
            # Preprocesar entrada
            processed_data = self.preprocess_input(application_data)
            
            # Realizar predicción
            prediction = self.model.predict(processed_data)[0]
            probability = self.model.predict_proba(processed_data)[0]
            
            # Determinar mensaje
            mensaje = "APROBADO" if prediction == 1 else "RECHAZADO"
            
            return {
                'prediccion': int(prediction),
                'probabilidad_aprobado': float(probability[1]),
                'probabilidad_rechazado': float(probability[0]),
                'confianza': float(max(probability)),
                'mensaje': mensaje
            }
        except Exception as e:
            raise ValueError(f"Error en predicción: {e}")
    
    def predict_batch(self, applications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predecir para múltiples solicitudes"""
        results = []
        errors = []
        
        for app_data in applications:
            try:
                result = self.predict_single(app_data)
                result['ID'] = app_data.get('ID', 'N/A')
                results.append(result)
            except Exception as e:
                errors.append(f"Error con ID {app_data.get('ID', 'N/A')}: {str(e)}")
        
        return results, errors