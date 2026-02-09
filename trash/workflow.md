### Complete Training Workflow
```mermaid
flowchart TD
    A[Start: python train.py] --> B[TrainPipeline.run_pipeline]
    B --> C[DataIngestion.initiate_data_ingestion]
    C --> D[Save artifacts/data.csv]
    C --> E[Train-Test Split 80/20]
    E --> F[Save artifacts/train.csv]
    E --> G[Save artifacts/test.csv]
    F & G --> H[DataTransformation.initiate_data_transformation]
    H --> I[Build Preprocessor Pipeline]
    I --> J[fit_transform on Train]
    I --> K[transform on Test]
    J & K --> L[Save artifacts/preprocessor.pkl]
    L --> M[ModelTrainer.initiate_model_trainer]
    M --> N[Train 8 Models + Hyperparameter Tuning]
    N --> O{Best R² > 0.6?}
    O -->|Yes| P[Save artifacts/model.pkl]
    O -->|No| Q[Raise CustomException]
    P --> R[Log Final R² Score]
    R --> S[Training Complete]
```

### Complete Inference Workflow
```mermaid
flowchart TD
    A[User Input] --> B{Interface?}
    B -->|Flask| C[flask_app.py /predictdata]
    B -->|Streamlit| D[streamlit_app.py Form]
    C --> E[CustomData.get_data_as_data_frame]
    D --> E
    E --> F[PredictPipeline.predict]
    F --> G[load_object preprocessor.pkl]
    F --> H[load_object model.pkl]
    G --> I[preprocessor.transform input]
    H --> J[model.predict features]
    I --> J
    J --> K[Return Prediction to UI]
    K --> L[User Sees Result]
```


```mermaid
flowchart TD
    subgraph A [Data Layer]
        A1[Data Sources<br>DBs / APIs / Files / Streams]
        A2[Data Lake / Warehouse<br>Raw Storage]
        A3[Feature
```


```mermaid
flowchart TD
    subgraph Data["Data Layer"]
        A1["notebooks/std.csv"]:::data
    end

    subgraph Training["Training Pipeline (train.py)"]
        B1["train_pipeline.py"]:::module
        B2["data_ingestion.py"]:::component
        B3["data_transformation.py"]:::component
        B4["model_trainer.py"]:::component
        B5["utils.py"]:::shared
    end

    subgraph Artifacts["Artifacts"]
        C1["data.csv, train.csv, test.csv, preprocessor.pkl, model.pkl"]:::artifact
    end
```