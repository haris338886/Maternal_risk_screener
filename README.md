# Maternal Risk Screener

Rural clinic decision-support tool for early maternal risk screening.
**Not a substitute for clinical judgment.**

## Setup (requires Python 3.12)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Train

```powershell
python -c "from src.train import train_and_save; train_and_save()"
```

## Run app

```powershell
streamlit run app/streamlit_app.py
```

## Test

```powershell
pytest tests/ -v
```

## Data

UCI Maternal Health Risk (ID=863), 1,013 samples, 6 vitals, 3-class target.