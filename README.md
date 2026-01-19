# **Irembo Voice AI: Intent Classification**

The goal of this project is to provide a reliable intent classification engine for government service access. The system is designed to handle low-resource language constraints, code-switching (Kinyarwanda/English/French), and noisy voice-to-text transcripts.

* **Core Model:** AfroXLMR-Small (Fine-tuned via LoRA).
* **Primary Metric:** Macro-F1 Score (ensures equitable performance across all intent classes).


## **System Architecture**

The project follows a modular MLOps design, separating data preparation, experimental training, and production serving.

1. **Orchestrator (`main.py`):** Acts as the central controller for the entire lifecycle.
2. **Training Pipeline:** Implements Stratified K-Fold Cross-Validation to find optimal hyperparameters before production retraining.
3. **Inference API:** A high-performance FastAPI service providing real-time intent predictions.


## **Data Strategy**

To handle the challenges of low-resource Kinyarwanda data and noisy transcripts,a dual-pronged strategy was implemented:

* **Data Augmentation**: Backtranslation generated 400+ more samples

* **Data Clinic**: Instead of a static cleaning step, out-of-sample predictions was used to audit label quality. Rows where the model is highly confident but disagrees with the human label were flagged and logged to `reports/potential_mislabels.csv` for review.


## **Installation & Setup**

### **Prerequisites**

* Python 3.10+

### **Local Setup**

```bash
# Clone the repository
git clone https://github.com/Joebasshd/irembo-voice-ai.git
cd irembo-voice-ai

# Create and activate virtual environment
python -m venv .venv
source .venv/Scripts/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

## **Execution Modes**

The system is controlled via `src/main.py`. Always run commands from the **root directory** using the module flag `-m`.

#### 1. Data Augmentation Mode

Expands minority intent classes using translation and paraphrasing.

```bash
python -m src.main augment --input data/voiceai_intent_train.csv

```

#### 2. Full Production Pipeline

Runs the end-to-end flow: Experiment (CV)  $\rightarrow$ Human-in-the-Loop Gate $\rightarrow$ Production Retrain $\rightarrow$ Evaluation $\rightarrow$ Inference.

```bash
python -m src.main pipeline --epochs 10 --batch_size 16

```

*Note: The script will pause after Phase 1 and ask for confirmation before building the final model.*

#### 3. Data Clinic Mode

Audits the latest training run to identify mislabeled data.

```bash
python -m src.main audit

```

### 4. Interactive Inference API

Launches the FastAPI server for real-time testing.

```bash
uvicorn src.inference:app --reload

```

Access the interactive documentation at: `http://127.0.0.1:8000/docs`.


## **Deployment with Docker**

For "environment parity" across development and production, use the provided Docker configuration.

```bash
# Build the image
docker build -t irembo-voice-ai .

# Run the container
docker run -p 8000:8000 -d irembo-voice-ai

```

Access the API:
Open `http://localhost:8000` in your browser.

## **Model Performance**
Macro F1 on test set: 0.9715

Macro F1 Score per Language

| Language    | F1 Score |
|------------|----------|
| Kinyarwanda | 0.9529   |
| English (en) | 1.0000   |
| Mixed       | 1.0000   |



## **Project Structure**

```text
irembo-voice-ai/
├── data/               # CSV datasets (Train/Val/Test)
├── models/             # LoRA adapters, Label Encoders, and Hyperparams
├── reports/            # Data Clinic audits and Eval metrics
├── src/                # Core Logic
│   ├── __init__.py     # Package marker
│   ├── main.py         # MLOps Controller
│   ├── train.py        # Experimental K-Fold CV logic
│   ├── inference.py    # FastAPI serving logic
│   ├── data_clinic.py  # Data auditing logic
│   └── eval_utils.py   # Model evaluation & metrics
├── Dockerfile          # Production containerization
└── requirements.txt    # Python dependencies

```

