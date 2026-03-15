from fastapi import FastAPI, HTTPException, Request
from typing import Dict, Any, Optional
from joblib import load
import numpy as np
import pandas as pd
import json
import os

MODEL_PATH = "realty_model.pkl"
FEATURES_PATH = "realty_feature_names.json"

# Минимальная цена (floor) берётся из метаданных, по умолчанию 1.0
MIN_PRICE_FLOOR = 1.0

app = FastAPI(title="Realty Prediction API", version="1.1.0")

# Проверки наличия файлов
if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError("Model or feature file not found. Run train_model.py first.")

# Загрузка модели и метаданных
model = load(MODEL_PATH)
with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

FEATURES = meta.get("features", [])
# Нормализация типа FEATURES: должно быть списком
if isinstance(FEATURES, str):
    FEATURES = [FEATURES]
elif not isinstance(FEATURES, (list, tuple)):
    try:
        FEATURES = list(FEATURES)
    except Exception:
        FEATURES = []

BASE_NUMERIC = meta.get("base_numeric_features", [])
CAT_FEATURES = meta.get("categorical_features", [])
TARGET = meta.get("target", "price")
LOG_TARGET = meta.get("log_target", False)  # True если в train_model.py использовался log1p
MIN_PRICE_FLOOR = float(meta.get("min_price", MIN_PRICE_FLOOR))

def to_period_num(period_str: Optional[str]) -> Optional[int]:
    if period_str is None:
        return None
    try:
        dt = pd.to_datetime(period_str, errors="coerce")
        if pd.isna(dt):
            return None
        return int(dt.year * 100 + dt.month)
    except Exception:
        return None

def _prepare_input_dataframe(input_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Создает DataFrame с исходными признаками (числовые и категориальные),
    без ручного кодирования. пайплайн внутри модели сделает кодирование.
    """
    # Числовые признаки
    numeric_vals = {}
    for feat in BASE_NUMERIC:
        if feat == 'period_num':
            # если есть период, конвертируем из 'period' в period_num
            v = input_dict.get('period')
            v_num = to_period_num(v)
            numeric_vals[feat] = float(v_num) if v_num is not None else 0.0
        else:
            v = input_dict.get(feat, 0.0)
            if v is None or (isinstance(v, str) and v.strip() == ""):
                v = 0.0
            try:
                numeric_vals[feat] = float(v)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail=f"Numeric feature '{feat}' must be a number.")

    # Категориальные признаки
    cat_vals = {}
    for c in CAT_FEATURES:
        val = input_dict.get(c, "")
        if val is None:
            val = ""
        cat_vals[c] = str(val)

    df_raw = pd.DataFrame([ {**numeric_vals, **cat_vals} ])
    return df_raw

def _predict_from_input(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    df_raw = _prepare_input_dataframe(input_dict)

    # Попытка привести вход к ожидаемому порядку колонок пайплайна
    try:
        df_to_predict = df_raw
        if hasattr(model, "named_steps"):
            preprocessor = None
            if "preprocessor" in model.named_steps:
                preprocessor = model.named_steps["preprocessor"]
            if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
                expected_cols = list(preprocessor.feature_names_in_)
                df_to_predict = df_raw.reindex(columns=expected_cols, fill_value=0)
    except Exception:
        df_to_predict = df_raw

    try:
        pred_raw = model.predict(df_to_predict)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    if LOG_TARGET:
        pred = np.expm1(pred_raw)
    else:
        pred = float(pred_raw)

    if pred < MIN_PRICE_FLOOR:
        pred = float(MIN_PRICE_FLOOR)

    return {
        "predicted_price": float(pred),
        "units": f"same as target '{TARGET}' in dataset",
        "log_target": bool(LOG_TARGET),
        "features_used": FEATURES
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/predict_get")
async def predict_get(request: Request):
    input_params = dict(request.query_params)
    try:
        res = _predict_from_input(input_params)
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_post")
async def predict_post(payload: Dict[str, Any]):
    try:
        res = _predict_from_input(payload)
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)