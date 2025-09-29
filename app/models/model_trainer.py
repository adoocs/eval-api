import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                           recall_score, f1_score, classification_report, 
                           roc_auc_score)
from sklearn.model_selection import GridSearchCV
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ModelTrainer:
    def __init__(self, model_path="models/random_forest_model.pkl"):
        self.model_path = model_path
        self.metadata_path = "models/model_metadata.json"
        self.results_path = "models/training_results.json"
        self.model = None
        self.feature_columns = None
        self.training_history = {}
        
        # Crear directorio models si no existe
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
    def load_data(self, csv_path):
        """Cargar y preparar los datos con análisis completo"""
        print("📊 Cargando dataset...")
        df = pd.read_csv(csv_path)
        
        # Análisis inicial del dataset
        print(f"✅ Dataset cargado: {len(df)} registros, {len(df.columns)} columnas")
        print(f"📈 Distribución de aprobados: {df['aprobado'].value_counts().to_dict()}")
        
        # Excluir columnas no features
        columns_to_drop = ['ID', 'aprobado']
        X = df.drop(columns_to_drop, axis=1, errors='ignore')
        y = df["aprobado"]
        
        # Análisis de variables
        self._analyze_features(X)
        
        # One-Hot Encoding
        print("🔧 Aplicando One-Hot Encoding...")
        X_encoded = pd.get_dummies(X)
        self.feature_columns = X_encoded.columns.tolist()
        
        print(f"🎯 Features generados: {len(self.feature_columns)} columnas")
        print("📋 Tipos de variables:")
        print(X_encoded.dtypes.value_counts())
        
        return X_encoded, y
    
    def _analyze_features(self, X):
        """Analizar características del dataset"""
        print("\n🔍 ANÁLISIS DE VARIABLES:")
        print(f"• Variables numéricas: {len(X.select_dtypes(include=[np.number]).columns)}")
        print(f"• Variables categóricas: {len(X.select_dtypes(include=['object']).columns)}")
        
        # Estadísticas básicas de variables numéricas
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print("\n📊 Estadísticas variables numéricas:")
            print(X[numeric_cols].describe())
    
    def optimize_hyperparameters(self, X_train, y_train):
        """Optimizar hiperparámetros del modelo"""
        print("\n🎯 OPTIMIZANDO HIPERPARÁMETROS...")
        
        # Parámetros para búsqueda (reducidos para velocidad)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced', None]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"✅ Mejores parámetros: {grid_search.best_params_}")
        print(f"✅ Mejor score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train(self, csv_path, test_size=0.2, random_state=42, optimize=False):
        """Entrenar el modelo con evaluación completa"""
        print("🚀 INICIANDO ENTRENAMIENTO DEL MODELO")
        print("=" * 60)
        
        # Cargar datos
        X, y = self.load_data(csv_path)
        
        # Dividir datos
        print(f"\n📊 Dividiendo datos: {test_size*100}% para test")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"• Training set: {X_train.shape[0]} muestras")
        print(f"• Test set: {X_test.shape[0]} muestras")
        
        # Entrenar modelo
        print("\n🤖 ENTRENANDO MODELO RANDOM FOREST...")
        start_time = datetime.now()
        
        if optimize:
            self.model = self.optimize_hyperparameters(X_train, y_train)
        else:
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
        
        training_time = datetime.now() - start_time
        print(f"✅ Entrenamiento completado en {training_time}")
        
        # Evaluación completa
        self._comprehensive_evaluation(X_train, X_test, y_train, y_test)
        
        # Validación cruzada
        self._cross_validation(X, y)
        
        # Análisis de importancia de features
        self._feature_analysis(X_train)
        
        # Guardar resultados
        self._save_training_results(X_test, y_test, training_time)
        
        return self.model
    
    def _comprehensive_evaluation(self, X_train, X_test, y_train, y_test):
        """Evaluación completa del modelo"""
        print("\n📈 EVALUACIÓN COMPLETA DEL MODELO")
        print("=" * 50)
        
        # Predicciones
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Métricas básicas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"✅ Exactitud (Accuracy): {accuracy:.4f}")
        print(f"✅ Precisión (Precision): {precision:.4f}")
        print(f"✅ Sensibilidad (Recall): {recall:.4f}")
        print(f"✅ F1-Score: {f1:.4f}")
        print(f"✅ AUC-ROC: {auc_roc:.4f}")
        
        # Matriz de confusión
        print("\n📊 MATRIZ DE CONFUSIÓN:")
        cm = confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix(cm)
        
        # Reporte de clasificación
        print("\n📋 REPORTE DE CLASIFICACIÓN:")
        print(classification_report(y_test, y_pred, target_names=['Rechazado', 'Aprobado']))
        
        # Métricas por clase
        tn, fp, fn, tp = cm.ravel()
        print("\n🔍 MÉTRICAS DETALLADAS:")
        print(f"• Verdaderos Negativos (TN): {tn}")
        print(f"• Falsos Positivos (FP): {fp}")
        print(f"• Falsos Negativos (FN): {fn}")
        print(f"• Verdaderos Positivos (TP): {tp}")
        print(f"• Tasa de Falsos Positivos: {fp/(fp+tn):.4f}")
        print(f"• Tasa de Falsos Negativos: {fn/(fn+tp):.4f}")
    
    def _cross_validation(self, X, y):
        """Validación cruzada del modelo"""
        print("\n🔄 VALIDACIÓN CRUZADA (5-fold)")
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='f1')
        
        print(f"✅ Scores de validación cruzada: {cv_scores}")
        print(f"✅ Media F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Añadir a historial
        self.training_history['cv_scores'] = cv_scores.tolist()
        self.training_history['cv_mean'] = cv_scores.mean()
        self.training_history['cv_std'] = cv_scores.std()
    
    def _feature_analysis(self, X_train):
        """Análisis de importancia de características"""
        print("\n🎯 ANÁLISIS DE IMPORTANCIA DE FEATURES")
        
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\n🏆 TOP 15 FEATURES MÁS IMPORTANTES:")
        print(feature_importance_df.head(15).to_string(index=False))
        
        # Plot de importancia
        self._plot_feature_importance(feature_importance_df.head(15))
        
        # Guardar importancia de features
        self.training_history['feature_importance'] = feature_importance_df.to_dict('records')
    
    def _plot_confusion_matrix(self, cm):
        """Graficar matriz de confusión"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Rechazado', 'Aprobado'],
                   yticklabels=['Rechazado', 'Aprobado'])
        plt.title('Matriz de Confusión')
        plt.ylabel('Real')
        plt.xlabel('Predicho')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Matriz de confusión guardada en: models/confusion_matrix.png")
    
    def _plot_feature_importance(self, importance_df):
        """Graficar importancia de features"""
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis', hue='feature')
        plt.title('Importancia de Features - Random Forest')
        plt.xlabel('Importancia')
        plt.tight_layout()
        plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Importancia de features guardada en: models/feature_importance.png")
    
    def _save_training_results(self, X_test, y_test, training_time):
        """Guardar resultados del entrenamiento"""
        # Predicciones finales
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        results = {
            'training_info': {
                'timestamp': datetime.now().isoformat(),
                'training_time_seconds': training_time.total_seconds(),
                'model_type': 'RandomForestClassifier',
                'test_set_size': len(X_test)
            },
            'performance_metrics': {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'auc_roc': roc_auc_score(y_test, y_pred_proba)
            },
            'model_parameters': self.model.get_params(),
            'training_history': self.training_history,
            'feature_columns_count': len(self.feature_columns)
        }
        
        with open(self.results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Resultados guardados en: {self.results_path}")
    
    def save_model(self):
        """Guardar el modelo y metadata con información completa"""
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado")
            
        # Guardar modelo
        joblib.dump(self.model, self.model_path)
        
        if os.path.exists(self.results_path):
            with open(self.results_path, 'r', encoding='utf-8') as f:
                training_results = json.load(f)
                performance_metrics = training_results.get('performance_metrics', {})
        else:
            performance_metrics = {}
            print("⚠️  No se encontraron resultados de entrenamiento")

        # Guardar metadata completa
        metadata = {
            'feature_columns': self.feature_columns,
            'model_type': 'RandomForestClassifier',
            'training_date': datetime.now().isoformat(),
            'model_parameters': self.model.get_params(),
            'feature_count': len(self.feature_columns),
            'classes': self.model.classes_.tolist(),
            'model_metrics': performance_metrics,
            'training_info': training_results.get('training_info', {}),
            'cv_scores': self.training_history.get('cv_scores', []),
            'cv_mean': self.training_history.get('cv_mean', 0)
        }
        
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Modelo guardado en: {self.model_path}")
        print(f"💾 Metadata guardada en: {self.metadata_path}")
        print(f"📊 Total de features: {len(self.feature_columns)}")
        
        return self.model_path
    
    def load_model(self):
        """Cargar modelo entrenado"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            self.feature_columns = metadata['feature_columns']
            print("✅ Modelo cargado exitosamente")
            return True
        else:
            print("❌ Modelo no encontrado")
            return False
