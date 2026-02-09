"""
Training Entry Point
====================
Single command to trigger complete ML pipeline.

Usage:
    python train.py

This script:
    - Instantiates TrainPipeline
    - Executes end-to-end training
    - Logs results to console and file
    - Exits with appropriate status code

Note: Artifacts are saved to artifacts/ directory
"""

from src.pipeline.train_pipeline import TrainPipeline
from src.logger import logging
import sys
import os


def main():
    """Main training execution function."""
    try:
        # ========================================
        # PRE-TRAINING CHECKS
        # ========================================
        print("\n" + "=" * 70)
        print("PRE-TRAINING VALIDATION")
        print("=" * 70)
        
        # Check if raw data exists
        raw_data_path = "notebook/data/WorkFromHomeBurnout.csv"
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(
                f"Raw data not found at: {raw_data_path}\n"
                "Please ensure dataset is placed in notebook/data/ directory."
            )
        
        print(f"Raw data found: {raw_data_path}")
        
        # Create artifacts directory if missing
        os.makedirs("artifacts", exist_ok=True)
        print(f"Artifacts directory ready: artifacts/")
        
        # ========================================
        # EXECUTE TRAINING PIPELINE
        # ========================================
        print("\n" + "=" * 70)
        print("STARTING MODEL TRAINING")
        print("=" * 70 + "\n")
        
        pipeline = TrainPipeline()
        results = pipeline.initiate_training()
        
        # ========================================
        # POST-TRAINING SUMMARY
        # ========================================
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"\n{'METRIC':<25} {'VALUE':<30}")
        print("-" * 70)
        print(f"{'R² Score':<25} {results['r2_score']:.4f}")
        print(f"{'Model Path':<25} {results['model_path']}")
        print(f"{'Preprocessor Path':<25} {results['preprocessor_path']}")
        print(f"{'Train Data':<25} {results['train_data_path']}")
        print(f"{'Test Data':<25} {results['test_data_path']}")
        print("=" * 70 + "\n")
        
        # Log success to file
        logging.info("TRAINING COMPLETED: Model and preprocessor saved successfully")
        
        return 0  # Success exit code
        
    except FileNotFoundError as fnf_error:
        print(f"\nFILE NOT FOUND ERROR:")
        print(f"   {str(fnf_error)}\n")
        logging.error(f"FileNotFoundError: {str(fnf_error)}")
        return 1
        
    except Exception as e:
        print(f"\nTRAINING FAILED:")
        print(f"   {str(e)}\n")
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)