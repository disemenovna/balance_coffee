#загружаем библиотеки
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *

sessions = data['session_duration'].values

#строим график врменного ряда
plt.figure(figsize=(22, 10))
ar = np.arange(len(sessions))
plt.plot(ar, sessions, 'r')
plt.show()

#строим график за первую неделю
sample = sessions[:168]
ar = np.arange(len(sample))
plt.figure(figsize=(22, 10))
plt.plot(ar, sample, 'r')
plt.show()

#подготовка данных
def prepare_data(seq, num):
    x, y = [], []
    for i in range(len(seq) - num):
        x.append(seq[i:i + num])
        y.append(seq[i + num])
    return np.array(x), np.array(y)

#указали длину последовательности - 2 недели или 336 часов
num = 48
x, y = prepare_data(sessions, num)

#делим датасет на обучающую и тестовую выборку
ind = int(0.8 * len(x))
x_tr, y_tr = x[:ind], y[:ind]
x_val, y_val = x[ind:], y[ind:]

#нормализуем данные
x_scaler = StandardScaler()
y_scaler = StandardScaler()

x_tr = x_scaler.fit_transform(x_tr)
x_val = x_scaler.transform(x_val)

y_tr = y_tr.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)

y_tr = y_scaler.fit_transform(y_tr).flatten()
y_val = y_scaler.transform(y_val).flatten()

#преобразуем даннные из двухмерного в трехмерный формат
x_tr = x_tr.reshape(x_tr.shape[0], x_tr.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

#компилируем модель
model = Sequential([
    LSTM(256, input_shape=(num, 1), activation='relu', return_sequences=True),
    Dropout(0.2),
    LSTM(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='linear'),
    Dense(1)
])
model.compile(loss=tf.keras.losses.MeanSquaredLogarithmicError(), optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredLogarithmicError()])

# @title Обучение модели
mc = ModelCheckpoint('best_model.keras', monitor='val_loss',
                      verbose=1, save_best_only=True, mode='min')

history = model.fit(x_tr, y_tr, epochs=50, batch_size=32,
                    validation_data=(x_val, y_val), callbacks=[mc])

model.save('model_koffi.h5')

#лучшая моделька
model.load_weights('best_model.keras')
mse = model.evaluate(x_val, y_val)
print("Mean Square Error:", mse)

#функция скользящего среднего
def compute_moving_average(data):
    return np.mean(data, axis=1)

#прогнозируем со скользящим средним
x_reshaped = x_val.reshape(-1, num)
y_pred = compute_moving_average(x_reshaped)
mse = np.mean((y_val - y_pred) ** 2)
print('Средний квадрат ошибки(скользящее среднее): ', mse)

#функция прогноза
def forecast(x_val, no_of_pred, ind):
    predictions = []
    temp = x_val[ind].copy()
    for _ in range(no_of_pred):
        pred = model.predict(temp.reshape(1, -1, 1), verbose=0)[0][0]
        predictions.append(pred)
        temp = np.append(temp, pred)[1:]
    return np.array(predictions)

#прогнозируем на 24 часа
ind = 12

no_of_pred_day = 24
y_pred_day = forecast(x_val, no_of_pred_day, ind)
y_true_day = y_val[ind:ind + no_of_pred_day]

no_of_pred_3day = 72
y_pred_3day = forecast(x_val, no_of_pred_3day, ind)
y_true_3day = y_val[ind:ind + no_of_pred_3day]

no_of_pred_week = 168
y_pred_week = forecast(x_val, no_of_pred_week, ind)
y_true_week = y_val[ind:ind + no_of_pred_week]

#обратно преобразовываем данные
y_true_day = y_scaler.inverse_transform(y_true_day.reshape(-1, 1)).flatten()
y_pred_day = y_scaler.inverse_transform(y_pred_day.reshape(-1, 1)).flatten()

y_true_3day = y_scaler.inverse_transform(y_true_3day.reshape(-1, 1)).flatten()
y_pred_3day = y_scaler.inverse_transform(y_pred_3day.reshape(-1, 1)).flatten()

y_true_week = y_scaler.inverse_transform(y_true_week.reshape(-1, 1)).flatten()
y_pred_week = y_scaler.inverse_transform(y_pred_week.reshape(-1, 1)).flatten()

#средняя длительность сеанса (в секундах)
average_session_duration = 600

#вычисляем общее количество пользователей за сутки
total_users_predicted_day = np.round(np.sum(y_pred_day) / average_session_duration)
total_users_actual_day = np.round(np.sum(y_true_day) / average_session_duration)

total_users_predicted_3day = np.round(np.sum(y_pred_3day) / average_session_duration)
total_users_actual_3day = np.round(np.sum(y_true_3day) / average_session_duration)

total_users_predicted_week = np.round(np.sum(y_pred_week) / average_session_duration)
total_users_actual_week = np.round(np.sum(y_true_week) / average_session_duration)

print(f'Общее количество пользователей за сутки (прогноз): {total_users_predicted_day:.2f}')
print(f'Общее количество пользователей за сутки (фактическое): {total_users_actual_day:.2f}')

print(f'Общее количество пользователей за 3 сутoк (прогноз): {total_users_predicted_3day:.2f}')
print(f'Общее количество пользователей за 3 суток (фактическое): {total_users_actual_3day:.2f}')


print(f'Общее количество пользователей за неделю (прогноз): {total_users_predicted_week:.2f}')
print(f'Общее количество пользователей за неделю (фактическое): {total_users_actual_week:.2f}')

#строим график
plt.figure(figsize=(22, 10))
ar = np.arange(len(y_true))
plt.plot(ar, y_true, 'r', label='Actual')
plt.plot(ar, y_pred, 'y', label='Predicted')
plt.legend()
plt.show()