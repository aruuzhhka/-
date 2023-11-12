import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
# Загрузка данных
file_path = r'C:/Users/Lenovo/Documents/forbes_2640_billionaires.csv'
data = pd.read_csv(file_path)
# Выбираем только нужные столбцы
selected_columns = ['net_worth', 'age', 'source', 'Education']
data = data[selected_columns]
# Заменяем возможные пустые значения
data.fillna(0, inplace=True)
# Преобразуем категориальные данные (source и Education) в числовой формат, используя One-Hot Encoding
data = pd.get_dummies(data)
# Разделение данных на признаки и целевую переменную
X = data.drop('net_worth', axis=1)
y = data['net_worth']
# Разделение данных на обучающий и тестовый набор
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Создание моделей
model1 = RandomForestRegressor()
model2 = GradientBoostingRegressor()
model3 = LinearRegression()
# Квазилинейная композиция (VotingRegressor)
ensemble = VotingRegressor(estimators=[('rf', model1), ('gb', model2), ('lr', model3)])
ensemble.fit(X_train, y_train)
# Оценка квазилинейной композиции
score = ensemble.score(X_test, y_test)
print(f'Коэффициент детерминации (R^2) для квазилинейной композиции: {score}')
