from app.models.model_trainer import ModelTrainer

def main():
    """Script para entrenar el modelo desde la línea de comandos"""
    try:
        trainer = ModelTrainer()
        accuracy = trainer.train("data/dataset_micro.csv")
        trainer.save_model()
        print(f"\n✅ Entrenamiento completado. Exactitud: {accuracy:.4f}")
    except Exception as e:
        print(f"❌ Error en entrenamiento: {e}")

if __name__ == "__main__":
    main()