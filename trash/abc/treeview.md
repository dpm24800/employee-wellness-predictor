


e2e-ml-project/
├── artifacts/          # Persistent outputs (data.csv, model.pkl, etc.)
├── logs/               # Timestamped execution logs
├── notebook/           # EDA & raw dataset (StudentsPerformance.csv)
├── src/
│   ├── components/     # Core ML logic (data_ingestion, transformation, model_trainer)
|   |   
│   ├── pipeline/       # Orchestration (train_pipeline, predict_pipeline)
│   ├── exception.py    # CustomException with stack trace
│   ├── logger.py       # Timestamped logging setup
│   ├── utils.py        # save_object/load_object helpers
│   └── __init__.py
├── templates/          # Flask HTML templates
├── train.py            # ⭐ SINGLE-COMMAND TRAINING TRIGGER
├── flask_app.py        # REST API for inference
├── streamlit_app.py    # Interactive UI for inference
├── requirements.txt
└── setup.py            # Package configuration
