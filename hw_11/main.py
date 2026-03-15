import os
from typing import Optional
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

MODEL_PATH = "realty_model.joblib"
DATA_CSV = "realty_data.csv"

# ---- Предобработка данных и обучение модели ----

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

def load_or_train_model(csv_path: str = DATA_CSV) -> Pipeline:
    if os.path.exists(MODEL_PATH):
        # Загружаем существующую модель
        clf = load(MODEL_PATH)
        return clf

    # Чтение данных
    df = pd.read_csv(csv_path)

    # Подготовка признаков: период -> period_num, drop price
    df = df.copy()
    df['period_num'] = df['period'].astype(str).apply(to_period_num)

    # Выбор признаков
    features = [
        'product_name', 'period_num', 'lat', 'lon',
        'object_type', 'total_square', 'rooms', 'floor', 'area',
        'city', 'settlement', 'district', 'postcode', 'source'
    ]
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    X = df[features]
    y = df['price']

    # Приведение некоторых типов (явная конвертация)
    if 'postcode' in X.columns:
        X['postcode'] = pd.to_numeric(X['postcode'], errors='coerce')

    # Приведение числовых признаков к числу и удаление явно некорректных значений
    numeric_features_all = ['period_num', 'lat', 'lon', 'total_square', 'rooms', 'floor', 'area', 'postcode']
    for col in numeric_features_all:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

    # Исключаем числовые признаки, которые состоят только из NaN
    numeric_features_present = []
    for col in numeric_features_all:
        if col in X.columns and not X[col].isna().all():
            numeric_features_present.append(col)

    if not numeric_features_present:
        raise ValueError("Нет числовых признаков с непустыми значениями для обучения. Проверьте данные.")

    # Разделяем на обучающую и валидационную выборки
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Препроцессинг: числовые признаки и категориальные признаки
    categorical_features = [
        'product_name', 'object_type', 'city', 'settlement', 'district', 'source'
    ]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features_present),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Модель
    model = RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1
    )

    clf = Pipeline(steps=[('preprocessor', preprocessor),
                         ('model', model)
                        ])

    # Обучение
    clf.fit(X_train, y_train)

    # Сохранение модели
    dump(clf, MODEL_PATH)

    return clf

# ---- Модель и предсказания ----

# Загрузка или обучение модели при старте приложения
model_pipeline = load_or_train_model()

class PredictionInput(BaseModel):
    product_name: Optional[str] = None
    period: Optional[str] = None
    postcode: Optional[str] = None
    address_name: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    object_type: Optional[str] = None
    total_square: Optional[float] = None
    rooms: Optional[float] = None
    floor: Optional[float] = None
    city: Optional[str] = None
    settlement: Optional[str] = None
    district: Optional[str] = None
    area: Optional[float] = None
    description: Optional[str] = None
    source: Optional[str] = None

def input_to_dataframe(p_input: PredictionInput) -> pd.DataFrame:
    # Преобразуем входные данные в одну строку датафрейма с нужными колонками
    row = {
        'product_name': p_input.product_name,
        'period_num': to_period_num(p_input.period),
        'lat': p_input.lat,
        'lon': p_input.lon,
        'object_type': p_input.object_type,
        'total_square': p_input.total_square,
        'rooms': p_input.rooms,
        'floor': p_input.floor,
        'area': p_input.area,
        'city': p_input.city,
        'settlement': p_input.settlement,
        'district': p_input.district,
        'postcode': p_input.postcode,
        'source': p_input.source
    }
    df = pd.DataFrame([row])
    # Привести postcode к числу, если возможно
    if 'postcode' in df.columns:
        df['postcode'] = pd.to_numeric(df['postcode'], errors='coerce')
    return df

def predict_price(p_input: PredictionInput) -> float:
    df = input_to_dataframe(p_input)
    pred = model_pipeline.predict(df)[0]
    return float(pred)

# ---- API ----

app = FastAPI(title="Realty Price Prediction API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/predict_get")
def predict_get(
    product_name: Optional[str] = None,
    period: Optional[str] = None,
    postcode: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    object_type: Optional[str] = None,
    total_square: Optional[float] = None,
    rooms: Optional[float] = None,
    floor: Optional[float] = None,
    city: Optional[str] = None,
    settlement: Optional[str] = None,
    district: Optional[str] = None,
    area: Optional[float] = None,
    source: Optional[str] = None
):
    input_model = PredictionInput(
        product_name=product_name,
        period=period,
        postcode=postcode,
        lat=lat,
        lon=lon,
        object_type=object_type,
        total_square=total_square,
        rooms=rooms,
        floor=floor,
        city=city,
        settlement=settlement,
        district=district,
        area=area,
        source=source
    )
    price = predict_price(input_model)
    return {"predicted_price": price}

@app.post("/predict_post")
def predict_post(input_model: PredictionInput):
    price = predict_price(input_model)
    return {"predicted_price": price}

# Чтобы запустить: uvicorn main:app --reload --port 8000
# Swagger доступен по http://127.0.0.1:8000/docs

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)