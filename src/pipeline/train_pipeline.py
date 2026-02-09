"""
TrainPipeline Module
====================
Orchestrates end-to-end training workflow:
Data Ingestion → Data Transformation → Model Training

Usage:
    from src.pipeline.train_pipeline import TrainPipeline
    
    pipeline = TrainPipeline()
    results = pipeline.initiate_training()
    
    print(f"R² Score: {results['r2_score']}")
    print(f"Model saved to: {results['model_path']}")
"""

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
import sys


class TrainPipeline:
    """
    Orchestrates the complete ML training pipeline.
    
    Responsibilities:
        - Sequences component execution (ingestion → transformation → training)
        - Manages artifact flow between components
        - Returns training metrics and artifact paths
        - Provides single entry point for reproducible training
    
    Attributes:
        data_ingestion (DataIngestion): Data ingestion component
        data_transformation (DataTransformation): Feature engineering component
        model_trainer (ModelTrainer): Model training component
    """
    
    def __init__(self):
        """Initialize pipeline with component instances."""
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
    
    def initiate_training(self):
        """
        Execute complete training pipeline.
        
        Workflow:
            1. Data Ingestion: Load raw data, rename columns, split train/test
            2. Data Transformation: Build preprocessing pipeline, transform features
            3. Model Training: Train 8 models, select best via GridSearchCV
        
        Returns:
            dict: Training results containing:
                - r2_score (float): Test R² score of best model
                - preprocessor_path (str): Path to saved preprocessor.pkl
                - model_path (str): Path to saved model.pkl
                - train_data_path (str): Path to train.csv
                - test_data_path (str): Path to test.csv
        
        Raises:
            CustomException: If any step fails (with full traceback)
        """
        try:
            logging.info("=" * 70)
            logging.info("🚀 INITIATING TRAINING PIPELINE")
            logging.info("=" * 70)
            
            # ========================================
            # STEP 1: DATA INGESTION
            # ========================================
            logging.info("\n[STEP 1/3] Data Ingestion")
            logging.info("-" * 70)
            
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()
            
            logging.info(f"✅ Data ingestion completed successfully")
            logging.info(f"   • Raw data: artifacts/data.csv")
            logging.info(f"   • Train set: {train_path}")
            logging.info(f"   • Test set: {test_path}")
            
            # ========================================
            # STEP 2: DATA TRANSFORMATION
            # ========================================
            logging.info("\n[STEP 2/3] Data Transformation")
            logging.info("-" * 70)
            
            train_arr, test_arr, preprocessor_path = \
                self.data_transformation.initiate_data_transformation(
                    train_path=train_path,
                    test_path=test_path
                )
            
            logging.info(f"✅ Data transformation completed successfully")
            logging.info(f"   • Preprocessor saved: {preprocessor_path}")
            logging.info(f"   • Train array shape: {train_arr.shape}")
            logging.info(f"   • Test array shape: {test_arr.shape}")
            
            # ========================================
            # STEP 3: MODEL TRAINING
            # ========================================
            logging.info("\n[STEP 3/3] Model Training")
            logging.info("-" * 70)
            
            r2_score = self.model_trainer.initiate_model_trainer(
                train_array=train_arr,
                test_array=test_arr
            )
            
            logging.info(f"✅ Model training completed successfully")
            logging.info(f"   • Best model R² score: {r2_score:.4f}")
            logging.info(f"   • Model saved: {self.model_trainer.model_trainer_config.trained_model_file_path}")
            
            # ========================================
            # PIPELINE COMPLETION
            # ========================================
            logging.info("\n" + "=" * 70)
            logging.info("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logging.info("=" * 70)
            
            return {
                "r2_score": r2_score,
                "preprocessor_path": preprocessor_path,
                "model_path": self.model_trainer.model_trainer_config.trained_model_file_path,
                "train_data_path": train_path,
                "test_data_path": test_path
            }
            
        except Exception as e:
            logging.error("❌ TRAINING PIPELINE FAILED", exc_info=True)
            raise CustomException(e, sys)


# ============================================================================
# STANDALONE EXECUTION (Optional: For direct testing)
# ============================================================================
if __name__ == "__main__":
    """
    Standalone execution for testing the pipeline.
    
    Usage:
        python -m src.pipeline.train_pipeline
    
    Note: For production training, use train.py (root directory) instead.
    """
    try:
        print("\n" + "=" * 70)
        print("🚀 STARTING TRAINING PIPELINE (Standalone Mode)")
        print("=" * 70 + "\n")
        
        pipeline = TrainPipeline()
        results = pipeline.initiate_training()
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"\n📊 RESULTS:")
        print(f"   • R² Score: {results['r2_score']:.4f}")
        print(f"   • Model: {results['model_path']}")
        print(f"   • Preprocessor: {results['preprocessor_path']}")
        print(f"   • Train Data: {results['train_data_path']}")
        print(f"   • Test Data: {results['test_data_path']}\n")
        
    except CustomException as ce:
        print(f"\n❌ Training failed with CustomException:")
        print(f"   {str(ce)}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during training:")
        print(f"   {str(e)}\n")
        sys.exit(1)