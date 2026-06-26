# SmartLab Analytics with EdgeSense AI

An autonomous, local-first platform for automating small-scale scientific and STEM research using Local LLMs and Edge AI.

## 📂 Project Directory Structure

Below is the complete architectural skeleton of the project. Please adhere to this layout when creating new modules or managing datasets.

```text
smartlab-analytics/
├── config/
│   └── app_config.json          # Global system settings and Ollama configurations
├── core/
│   ├── __init__.py
│   ├── edge_dsp.py              # Edge-level processing (smoothing, moving average, RMS)
│   ├── feature_extractor.py     # Time-series analysis and feature extraction using tsfresh
│   └── llm_analyzer.py          # Integration with Local LLM (Ollama)
├── data/
│   └── Projects/                # Main repository grouped by research projects
│       ├── Experiment_Thermodynamics_01/
│       │   ├── project_metadata.json   # General info about this experiment (Goal, Date, Author)
│       │   └── Sensors/                # Sensors isolated inside this specific project
│       │       ├── Sensor_Temperature/
│       │       │   ├── metadata.json   # Sensor calibration and type profiles
│       │       │   ├── 2026-06-25.csv  # Daily log files (Timestamp, Value)
│       │       │   └── 2026-06-26.csv
│       │       └── Sensor_Pressure/
│       │           ├── metadata.json
│       │           └── 2026-06-26.csv
│       │
│       ├── Experiment_Biophysics_PlantGrowth/
│       │   ├── project_metadata.json
│       │   └── Sensors/
│       │       ├── Sensor_CO2/
│       │       │   ├── metadata.json
│       │       │   └── 2026-06-26.csv
│       │       └── Sensor_Humidity/
│       │           ├── metadata.json
│       │           └── 2026-06-26.csv
├── dashboard/
│   └── app.py                   # UI with Project Selection -> Sensor Selection views
├── requirements.txt             # Project dependencies (pandas, tsfresh, ollama, streamlit)
└── main.py                      # Main entry point (expects project_id and sensor_id as arguments)

## 🛠️ Documentation

Detailed explanations of the internal mechanics, digital signal processing (DSP), tsfresh features, and Local LLM prompt strategies can be found in the ARCHITECTURE.md file.