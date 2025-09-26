from datetime import datetime
from app.models.predictor import LoanPredictor

class UnifiedPredictor:
    def __init__(self):
        self.predictor = LoanPredictor()
    
    def calcular_capacidad_pago(self, datos):
        """Calcular capacidad de pago según criterio empresa"""
        ingresos_totales = (datos['ingreso_mensual_neto'] + 
                           datos.get('ingreso_adicional', 0) + 
                           datos.get('ingreso_terceros', 0))
        
        gastos_totales = (datos['gasto_basico'] + 
                         datos.get('gastos_operativos', 0) + 
                         datos.get('gastos_financieros', 0))
        
        if ingresos_totales > 0:
            return ((ingresos_totales - gastos_totales) / ingresos_totales) * 30 / 100
        return 0
    
    def clasificar_riesgo(self, capacidad_pago):
        """Clasificar riesgo según criterio empresa"""
        if capacidad_pago > 75:
            return "CRÍTICO", "DENEGACIÓN AUTOMÁTICA", "EXCEDE 75%"
        elif capacidad_pago > 50:
            return "ALTO", "RECOMENDACIÓN RECHAZAR", "50%-75%"
        elif capacidad_pago > 25:
            return "MODERADO", "EVALUAR OTROS FACTORES", "25%-50%"
        else:
            return "BAJO", "APROBABLE", "MENOR 25%"
    
    def calcular_metricas_riesgo(self, datos, capacidad_pago):
        """Calcular múltiples métricas de riesgo"""
        # Ratio de endeudamiento
        ingresos_totales = datos['ingreso_mensual_neto'] + datos.get('ingreso_adicional', 0)
        ratio_endeudamiento = (datos.get('gastos_financieros', 0) / ingresos_totales * 100) if ingresos_totales > 0 else 0
        
        # Relación deuda/ingreso
        relacion_deuda_ingreso = (datos.get('deuda_total_financiera', 0) / datos['ingreso_mensual_neto'] * 100) if datos['ingreso_mensual_neto'] > 0 else 0
        
        # Margen de seguridad
        margen_seguridad = 100 - capacidad_pago
        
        return {
            "capacidad_pago_porcentaje": capacidad_pago,
            "ratio_endeudamiento": round(ratio_endeudamiento, 2),
            "relacion_deuda_ingreso": round(relacion_deuda_ingreso, 2),
            "margen_seguridad": round(margen_seguridad, 2),
            "ingresos_totales": ingresos_totales,
            "gastos_totales": datos['gasto_basico'] + datos.get('gastos_operativos', 0) + datos.get('gastos_financieros', 0)
        }
    
    def predecir_completo(self, datos_solicitud):
        """Predicción unificada con todas las métricas"""
        # 1. Predicción ML
        resultado_ml = self.predictor.predict_single(datos_solicitud)
        
        # 2. Análisis de riesgo
        capacidad_pago = self.calcular_capacidad_pago(datos_solicitud)
        nivel_riesgo, recomendacion_riesgo, criterio_empresa = self.clasificar_riesgo(capacidad_pago)
        metricas_riesgo = self.calcular_metricas_riesgo(datos_solicitud, capacidad_pago)
        
        # 3. Variables clave para análisis
        variables_clave = {
            "puntaje_experian": datos_solicitud.get('puntaje_experian', 0),
            "monto_solicitado": datos_solicitud.get('monto_solicitado', 0),
            "ingreso_mensual_neto": datos_solicitud.get('ingreso_mensual_neto', 0),
            "plazo_solicitado": datos_solicitud.get('plazo', 0)
        }
        
        # 4. Decisión final integrada
        if nivel_riesgo == "CRÍTICO":
            decision_final = "RECHAZADO"
            motivo = f"Capacidad de pago insuficiente: {capacidad_pago:.1f}% (Criterio: {criterio_empresa})"
        elif resultado_ml['prediccion'] == 1 and nivel_riesgo in ["BAJO", "MODERADO"]:
            decision_final = "APROBADO"
            motivo = f"Aprueba ML ({resultado_ml['confianza']:.1%}) y riesgo {nivel_riesgo}"
        else:
            decision_final = "RECHAZADO"
            motivo = f"ML: {resultado_ml['mensaje']}, Riesgo: {nivel_riesgo}"
        
        # 5. Respuesta unificada
        return {
            "ID": datos_solicitud.get('ID', 'N/A'),
            "timestamp": datetime.now().isoformat(),
            
            # ML
            "prediccion_ml": resultado_ml['prediccion'],
            "probabilidad_aprobado": resultado_ml['probabilidad_aprobado'],
            "probabilidad_rechazado": resultado_ml['probabilidad_rechazado'],
            "confianza_ml": resultado_ml['confianza'],
            "mensaje": resultado_ml['mensaje'],
            
            # Riesgo
            "capacidad_pago": round(capacidad_pago, 2),
            "nivel_riesgo": nivel_riesgo,
            "recomendacion_riesgo": recomendacion_riesgo,
            "criterio_empresa": criterio_empresa,
            
            # Métricas
            "metricas_riesgo": metricas_riesgo,
            "variables_clave": variables_clave,
            
            # Decisión
            "decision_final": decision_final,
            "motivo": motivo
        }