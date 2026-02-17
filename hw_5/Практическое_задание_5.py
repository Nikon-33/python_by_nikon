# =========================== #
# Полярс: работа с train.csv
import polars as pl

# 1. Считать датасет
df_poly = pl.read_csv('train.csv')
print(dir(df_poly))
# 2. Основная информация о датасете
print("Типы данных:\n", df_poly.dtypes)
print("Количество пропусков:\n", df_poly.null_count())
print("Описание:\n", df_poly.describe())
print("Средние значения:\n", df_poly.select([pl.all().exclude(pl.Utf8).mean()]))

# 3. Количество пассажиров каждого класса
pclass_counts = df_poly.get_column('Pclass').value_counts()
print("Количество по классам:\n", pclass_counts)

# 4. Количество выживших мужчин и женщин
survivors_by_gender = df_poly.group_by('Sex').agg(
    pl.col('Survived').sum().alias('Survivors')
)
print("Выжившие по полу:\n", survivors_by_gender)

# 5. Таблица пассажиров старше 44 лет
older_passengers = df_poly.filter(pl.col('Age') > 44)
print("Пассажиры старше 44:\n", older_passengers)


# ================================================================================== #
# ускорение и расчет
import pandas as pd
import bottleneck as bn

df_pd = pd.read_csv('train.csv')

# 2. Средний возраст и стандартное отклонение с помощью bottleneck
mean_age = bn.nanmean(df_pd['Age'])
std_age = bn.nanstd(df_pd['Age'])
print(f"Средний возраст: {mean_age}")
print(f"Стандартное отклонение: {std_age}")

# 3. Умножение Fare на 1.3
df_pd['Fare_new'] = df_pd['Fare'].apply(lambda x: x * 1.3)

# Альтернатива (через itertuples) для большей производительности:
# for row in df_pd.itertuples():
#     df_pd.at[row.Index, 'Fare_new'] = row.Fare * 1.3


# ================================================================================== #
# оптимизация типов
df_housing = pd.read_csv('Housing.csv')

# 2. Анализ типов и памяти
for col in df_housing.columns:
    print(f"{col} -> {df_housing[col].dtype}")
    print(f"Память: {df_housing[col].memory_usage(deep=True)} байт\n")

# print(df_housing.columns)

print("Общая память до:")
print(df_housing.memory_usage(deep=True).sum())

# 3. Замена типов данных для экономии памяти
df_housing['bedrooms'] = df_housing['bedrooms'].astype('int16')
df_housing['price'] = df_housing['price'].astype('float32')

# После смены типов:
print("Общая память после:")
print(df_housing.memory_usage(deep=True).sum())
