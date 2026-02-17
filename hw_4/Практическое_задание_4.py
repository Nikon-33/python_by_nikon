import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_csv('train.csv')

plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title('Распределение выживаемости (Survived)')
plt.xlabel('Выжили (0 = Нет, 1 = Да)')
plt.ylabel('Количество')
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', data=df)
plt.title('Распределение классов (Pclass)')
plt.xlabel('Класс (1, 2, 3)')
plt.ylabel('Количество')
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['Age'].dropna(), bins=30)
plt.title('Распределение возраста (Age)')
plt.xlabel('Возраст')
plt.ylabel('Количество')
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='Sex', data=df)
plt.title('Распределение по полу (Sex)')
plt.xlabel('Пол')
plt.ylabel('Количество')
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='Parch', data=df)
plt.title('Распределение Parch (количество родственников рядом)')
plt.xlabel('Parch')
plt.ylabel('Количество')
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x='Age', data=df)
plt.title('Boxplot для возраста (Age)')
plt.xlabel('Возраст')
plt.show()

survived_counts = df['Survived'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(survived_counts,
        labels=['Не выжили', 'Выжили'],
        autopct='%1.1f%%',
        colors=['lightblue', 'lightgreen'])
plt.title('Доля пассажиров по выживаемости')
plt.show()

pclass_counts = df['Pclass'].value_counts().sort_index()
plt.figure(figsize=(6,6))
plt.pie(pclass_counts,
        labels=['Класс 1', 'Класс 2', 'Класс 3'],
        autopct='%1.1f%%',
        colors=['gold', 'silver', 'brown'])
plt.title('Доля пассажиров по классам (Pclass)')
plt.show()

sns.pairplot(df, vars=['Age', 'Fare', 'SibSp', 'Parch'], hue='Survived')
plt.suptitle('Pairplot числовых переменных', y=1.02)
plt.show()

sunburst_df = df.groupby(['Pclass', 'Sex']).size().reset_index(name='Count')

fig = px.sunburst(
    sunburst_df,
    path=['Pclass', 'Sex'],
    values='Count',
    title='Sunburst: Пассажиры по классу и полу'
)
fig.show()