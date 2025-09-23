import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json
import os

class ModelTrainer:
    def __init__(self, model_path="models/random_forest_model.pkl"):
        self.model_path = model_path
        self.metadata_path = "models/model_metadata.json"
        self.model = None
        self.feature_columns = None
        
        # Crear directorio models si no existe
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
    def load_data(self, csv_path):
        """Cargar y preparar los datos"""
        df = pd.read_csv(csv_path)
        columns_to_drop = ['ID', 'aprobado']  # Asegurar que ID se excluya
        X = df.drop(columns_to_drop, axis=1)
        y = df["aprobado"]
        
        # One-Hot Encoding
        X_encoded = pd.get_dummies(X)
        self.feature_columns = X_encoded.columns.tolist()
        
        return X_encoded, y
    
    def train(self, csv_path, test_size=0.3, random_state=42):
        """Entrenar el modelo"""
        print("Cargando datos...")
        X, y = self.load_data(csv_path)
        
        print("Dividiendo en train/test...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print("Entrenando modelo...")
        self.model = RandomForestClassifier(
            n_estimators=200, 
            class_weight="balanced", 
            random_state=random_state
        )
        self.model.fit(X_train, y_train)
        
        # Evaluación
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n=== EVALUACIÓN DEL MODELO ===")
        print("Matriz de confusión:")
        print(confusion_matrix(y_test, y_pred))
        print(f"\nExactitud: {accuracy:.4f}")
        
        # Importancia de variables
        importances = pd.Series(
            self.model.feature_importances_, 
            index=X.columns
        ).sort_values(ascending=False)
        print("\nTop 10 variables más importantes:")
        print(importances.head(10))
        
        return accuracy
    
    def save_model(self):
        """Guardar el modelo y metadata"""
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado")
            
        # Guardar modelo
        joblib.dump(self.model, self.model_path)
        
        # Guardar metadata (columnas)
        metadata = {
            'feature_columns': self.feature_columns,
            'model_type': 'RandomForestClassifier'
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Modelo guardado en: {self.model_path}")
        return self.model_path