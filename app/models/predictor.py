import time
import pandas as pd
import joblib
import json
import os
from typing import Dict, Any, List, Tuple

class LoanPredictor:
    def __init__(self, model_path="models/random_forest_model.pkl", 
                 metadata_path="models/model_metadata.json",
                 risk_threshold=0.6):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.risk_threshold = risk_threshold
        self.model = None
        self.feature_columns = None
        self.is_loaded = False
        self.load_model()
    
    def load_model(self):
        """Cargar modelo y metadata con mejor manejo de errores"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modelo no encontrado en: {self.model_path}")
            
            self.model = joblib.load(self.model_path)
            
            if not os.path.exists(self.metadata_path):
                raise FileNotFoundError(f"Metadata no encontrada en: {self.metadata_path}")
                
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
            self.feature_columns = metadata.get('feature_columns', [])
            self.model_metrics = metadata.get('model_metrics', {})
            
            if not self.feature_columns:
                print("⚠️  Advertencia: No se encontraron feature_columns en metadata")
                
            self.is_loaded = True
            print("✅ Modelo cargado exitosamente")
            print(f"📊 Métricas del modelo: {self.model_metrics}")
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            self.is_loaded = False
    
    def calculate_risk_metrics(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        """Calcular métricas de riesgo con límites aplicados"""
        try:
            # Obtener valores
            ingresos_totales = input_data.get('ingreso_total', 0)
            gastos_totales = input_data.get('gasto_total', 0)
            deuda_total = input_data.get('gastos_financieros', 0)
            puntaje_experian = input_data.get('puntaje_experian', 0)

            # Calcular métricas base
            if ingresos_totales <= 0:
                capacidad_pago = -100
                indice_endeudamiento = 500  # Límite máximo
            else:
                capacidad_pago = ((ingresos_totales - gastos_totales) / ingresos_totales) * 100
                indice_endeudamiento = (deuda_total / ingresos_totales) * 100

            ajuste_historial = (100 - (puntaje_experian / 10)) * 0.4

            # Calcular riesgo crediticio
            riesgo_crediticio = (
                (indice_endeudamiento * 0.4) + 
                ((1 - (capacidad_pago / 100)) * 100 * 0.4) + 
                ajuste_historial
            )

            # ✅ APLICAR LÍMITES según tus modelos Pydantic
            return {
                'capacidad_pago_porcentaje': self._limitar_valor(capacidad_pago, -100, 100),
                'indice_endeudamiento': self._limitar_valor(indice_endeudamiento, 0, 500),
                'ajuste_historial': self._limitar_valor(ajuste_historial, 0, 40),
                'riesgo_crediticio': self._limitar_valor(riesgo_crediticio, 0, 100)
            }

        except Exception as e:
            print(f"⚠️  Error calculando métricas: {e}")

    def _limitar_valor(self, valor: float, minimo: float, maximo: float) -> float:
        """Limitar un valor entre un mínimo y máximo"""
        valor_limitado = max(minimo, min(valor, maximo))
        return round(valor_limitado, 2)
    
    def preprocess_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocesar los datos de entrada MEJORADO"""
        try:
            # Crear copia para no modificar el original
            data_copy = input_data.copy()

            # Extraer campos no features
            data_copy.pop('ID', 'N/A')
            data_copy.pop('aprobado', None)

            # Crear DataFrame y aplicar one-hot encoding
            df = pd.DataFrame([data_copy])
            df_encoded = pd.get_dummies(df)

            # Asegurar todas las columnas esperadas
            if self.feature_columns:
                missing_cols = set(self.feature_columns) - set(df_encoded.columns)
                for col in missing_cols:
                    df_encoded[col] = 0
                
                # Reordenar columnas
                df_encoded = df_encoded.reindex(columns=self.feature_columns, fill_value=0)
            else:
                print("⚠️  Usando columnas generadas automáticamente")
                self.feature_columns = df_encoded.columns.tolist()

            return df_encoded
            
        except Exception as e:
            raise ValueError(f"Error en preprocesamiento: {e}")
    
    def get_decision_reason(self, prediction: int, probabilities: list, risk_metrics: dict) -> str:
        """Generar razón de la decisión para mayor transparencia"""
        prob_aprobado = probabilities[1]
        
        if prediction == 1:
            if prob_aprobado > 0.8:
                return "Alta capacidad de pago y buen perfil crediticio"
            elif risk_metrics.get('riesgo_crediticio', 100) < 50:
                return "Riesgo crediticio dentro de límites aceptables"
            else:
                return "Aprobado con observaciones - perfil moderado"
        else:
            if prob_aprobado < 0.3:
                return "Baja probabilidad de pago"
            elif risk_metrics.get('capacidad_pago_porcentaje', 0) < 10:
                return "Capacidad de pago insuficiente"
            elif risk_metrics.get('indice_endeudamiento', 100) > 70:
                return "Nivel de endeudamiento muy alto"
            else:
                return "Perfil no cumple con los criterios mínimos"
    
    def predict_single(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Realizar predicción para una sola solicitud MEJORADO"""
        if not self.is_loaded:
            raise ValueError("Modelo no cargado. Entrene el modelo primero.")
        
        try:
            # ⏱️ INICIAR CRONÓMETRO
            start_time = time.time()

            # Preprocesar entrada
            processed_data = self.preprocess_input(application_data)
            
            # Realizar predicción
            prediction = self.model.predict(processed_data)[0]
            probability = self.model.predict_proba(processed_data)[0]
            
            # Calcular métricas de riesgo para tu tesis
            risk_metrics = self.calculate_risk_metrics(application_data)
            
            # Determinar mensaje y razón
            prob_aprobado = float(probability[1])
            mensaje = "APROBADO" if prediction == 1 else "RECHAZADO"
            razon = self.get_decision_reason(prediction, probability, risk_metrics)
            
            # ⏱️ CALCULAR TIEMPO DE PROCESAMIENTO
            processing_time = time.time() - start_time

            return {
                'prediccion': int(prediction),
                'probabilidad_aprobado': prob_aprobado,
                'probabilidad_rechazado': float(probability[0]),
                'confianza': float(max(probability)),
                'mensaje': mensaje,
                'razon': razon,
                'metricas_riesgo': risk_metrics,
                'modelo_metricas': self.model_metrics,
                'tiempo_procesamiento_segundos': round(processing_time, 4),
                'tiempo_procesamiento_ms': round(processing_time * 1000, 2)
            }
            
        except Exception as e:
            raise ValueError(f"Error en predicción: {e}")
    
    def predict_batch(self, applications: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Predecir para múltiples solicitudes MEJORADO"""
        results = []
        errors = []
        
        for i, app_data in enumerate(applications):
            try:
                result = self.predict_single(app_data)
                result['ID'] = app_data.get('ID', f'UNKNOWN_{i}')
                results.append(result)
            except Exception as e:
                app_id = app_data.get('ID', f'UNKNOWN_{i}')
                error_msg = f"Error con ID {app_id}: {str(e)}"
                errors.append(error_msg)
                print(f"❌ {error_msg}")
        
        print(f"✅ Procesadas {len(results)} solicitudes, {len(errors)} errores")
        return results, errors

    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo para tu tesis"""
        return {
            'cargado': self.is_loaded,
            'caracteristicas': len(self.feature_columns) if self.feature_columns else 0,
            'metricas': self.model_metrics,
            'umbral_riesgo': self.risk_threshold
        }