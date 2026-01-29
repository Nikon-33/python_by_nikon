import pandas as pd

('1. Читаем датасет из файла train.csv')
df = pd.read_csv('train.csv')
print("Датасет загружен:", df.shape)

print('2. Выводим основную информацию о датасете')
print("\nИнформация о датасете:")
print(df.info())

print("\nСтатистики по датасету:")
print(df.describe(include='all'))

# Создаем колонку с именами пассажиров
def extract_first_name(name):
    if pd.isnull(name):
        return ''
    parts = name.split(',')
    if len(parts) > 1:
        first_part = parts[1]
        first_name = first_part.strip().split(' ')[-1]
        return first_name
    return ''

# Добавляем колонку с именем
df['FirstName'] = df['Name'].apply(extract_first_name)

print('\n3. Считаем процент выживаемости у каждого класса пассажиров')
survival_by_class = df.groupby('Pclass')['Survived'].mean() * 100
print("Процент выживаемости по классам (в %):")
print(survival_by_class)

print('\n4. Выводим самое популярное мужское и женское имя на корабле')
most_common_male_name = df[df['Sex']=='male']['FirstName'].mode()[0]
most_common_female_name = df[df['Sex']=='female']['FirstName'].mode()[0]
print(f"Самое популярное мужское имя: {most_common_male_name}")
print(f"Самое популярное женское имя: {most_common_female_name}")

print('\n5. Выводим самое популярное мужское и женское имя в каждом классе')
popular_names_per_class = df.groupby(['Pclass', 'Sex'])['FirstName'].agg(lambda x: x.mode()[0])
print("Самое популярное имя в каждом классе и поле:")
print(popular_names_per_class)

print('\n6. Таблица с пассажирами, возраст которых больше 44 лет')
older_than_44 = df[df['Age'] > 44]
print("Пассажиры старше 44 лет:")
print(older_than_44)

print('\n7. Таблица с пассажирами, возраст которых меньше 44 лет и которые мужского пола')
males_older_than_44 = df[(df['Age'] < 44) & (df['Sex'] == 'male')]
print("Мужчины младше 44 лет:")
print(males_older_than_44)

print('\n8. Количество n-местных кабин (в которых было 2,3,4,... человек)')
cabins = df['Cabin'].dropna()

cabin_counts = cabins.value_counts()
print("Количество пассажиров в каждом кабинете:")
print(cabin_counts)