import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.dates as mdates

# Carrega e processa dataset
variables = pd.read_csv('../data/dataset.csv', index_col=0, parse_dates=True)
scaler_ws100 = MinMaxScaler()
scaler_other = MinMaxScaler()

variables_scaled = variables.copy()
ws100_columns = ['ws100_wavelet']
other_columns = variables.columns.drop('ws100_wavelet').tolist()
variables_scaled[ws100_columns] = scaler_ws100.fit_transform(variables[ws100_columns])
variables_scaled[other_columns] = scaler_other.fit_transform(variables[other_columns])

input_steps = 72
output_steps = 36
data_array = variables_scaled.values
target_col_index = variables_scaled.columns.get_loc('ws100_wavelet')

def create_sequences(data, input_steps, output_steps, target_col_index):
    X_encoder = []
    X_decoder = []
    y_decoder = []
    for i in range(len(data) - input_steps - output_steps + 1):
        X_encoder.append(data[i:(i + input_steps)])
        decoder_input = np.zeros((output_steps, 1))
        decoder_input[0] = data[i + input_steps - 1, target_col_index]
        decoder_input[1:] = data[i + input_steps:i + input_steps + output_steps - 1, target_col_index].reshape(-1, 1)
        X_decoder.append(decoder_input)
        y_decoder.append(data[i + input_steps:i + input_steps + output_steps, target_col_index].reshape(-1, 1))
    return np.array(X_encoder), np.array(X_decoder), np.array(y_decoder)

X_encoder, X_decoder, y_decoder = create_sequences(data_array, input_steps, output_steps, target_col_index)

def rolling_forecasting_real_time_improved(model, data_scaled, input_steps, output_steps, scaler_ws100, target_col_index):
    """
    Realiza rolling forecasting simulando um cenário de previsão em tempo real com feedback do decoder.
    
    :param model: Modelo treinado.
    :param data_scaled: Dados escalonados (inclui treino, validação e teste).
    :param input_steps: Número de passos de entrada.
    :param output_steps: Número de passos de saída.
    :param scaler_ws100: Escalador para a coluna 'ws100_wavelet'.
    :param target_col_index: Índice da coluna alvo.
    :return: Arrays com previsões e valores reais, e o índice de início do teste.
    """
    predictions = []
    actuals = []
    
    test_start = int(data_scaled.shape[0] * 0.95) 
    test_data = data_scaled[test_start:]
    
    window_start = test_start - input_steps
    window_end = test_start
    window = data_scaled[window_start:window_end].copy()
    
    for i in range(len(test_data) - output_steps + 1):
        encoder_input = window.reshape(1, input_steps, data_scaled.shape[1])
        decoder_input = np.zeros((1, output_steps, 1))
        decoder_input[0, 0, 0] = encoder_input[0, -1, target_col_index]
        
        for t in range(1, output_steps):
            previous_pred = decoder_input[0, t-1, 0]
            decoder_input[0, t, 0] = previous_pred
        
        pred = model.predict([encoder_input, decoder_input], verbose=0)
        
        last_pred = pred[0, -1, 0]
        predictions.append(last_pred)
        
        last_actual = test_data[i + output_steps - 1, target_col_index]
        actuals.append(last_actual)
        
        window = np.vstack([window, test_data[i + output_steps - 1]])
        window = window[1:]
    
    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)
    
    predictions_inverse = scaler_ws100.inverse_transform(predictions)
    actuals_inverse = scaler_ws100.inverse_transform(actuals)
    
    return predictions_inverse, actuals_inverse, test_start

# Carregar o modelo treinado
model = load_model('../models/best_model.h5.keras')

# Executar a previsão rolling
predictions, actuals, test_start = rolling_forecasting_real_time_improved(
    model, 
    data_scaled=variables_scaled.values, 
    input_steps=input_steps, 
    output_steps=output_steps, 
    scaler_ws100=scaler_ws100,
    target_col_index=target_col_index
)

offset = 36
current_index = 0
step = 2

len_test = len(actuals)
start_time_idx = test_start + output_steps - 1
end_time_idx = start_time_idx + len_test
full_time_index = variables.index[start_time_idx:end_time_idx]

# Carregando os dados brutos para comparação
data = pd.read_csv("../data/dataset.csv")
data['id_str'] = data['id'].astype(str).apply(lambda x: x.split('.')[0])
data['id_datetime'] = pd.to_datetime(data['id_str'], errors='coerce')
data.set_index('id_datetime', inplace=True)
data.drop(columns=['id_str'], inplace=True)

# Extraindo a série ws100 original do mesmo intervalo de tempo
ws100_values = data.loc[full_time_index, 'ws100'].values

# Configuração dos gráficos
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(16,9))
plt.subplots_adjust(bottom=0.2, hspace=0.7)

# Linha de referência a 4.5 m/s (limite operacional)
ax.axhline(y=4.5, color='green', linestyle='--')
line_actual, = ax.plot([], [], label='Real (denoised)', color='blue')
line_pred, = ax.plot([], [], label='Previsto', color='red')

ax.legend(loc='upper left')
ax.set_xlabel('Tempo (Data/Hora)')
ax.set_ylabel('ws100 (m/s)')
ax.set_title('Simulação de Previsão (Denoised)')
error_text = ax.text(0.15, 0.9, '', transform=ax.transAxes)

# Segundo gráfico (comparando data['ws100'] com prediction)
ax2.axhline(y=4.5, color='green', linestyle='--')
line_actual2, = ax2.plot([], [], label='Real (Bruto)', color='blue')
line_pred2, = ax2.plot([], [], label='Previsto', color='red')

ax2.legend(loc='upper left')
ax2.set_xlabel('Tempo (Data/Hora)')
ax2.set_ylabel('ws100 (m/s)')
ax2.set_title('Simulação de Previsão (Bruto)')
error_text2 = ax2.text(0.15, 0.9, '', transform=ax2.transAxes)

# Configurações de formatação do eixo x
hours_to_mark = [0,6,12,18]
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=hours_to_mark))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H'))
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

ax2.xaxis.set_major_locator(mdates.HourLocator(byhour=hours_to_mark))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H'))
plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')

def update_plot():
    """Atualiza os gráficos com os dados até o índice atual"""
    max_len = len(actuals)
    end_act = min(current_index, max_len)
    end_pred = min(offset + current_index, max_len)
    
    # Primeiro gráfico (actuals vs predictions)
    time_actual = full_time_index[:end_act]
    time_pred = full_time_index[:end_pred]
    
    y_actual = actuals[:end_act].reshape(-1)
    y_pred = predictions[:end_pred].reshape(-1)
    
    line_actual.set_xdata(time_actual)
    line_actual.set_ydata(y_actual)
    line_pred.set_xdata(time_pred)
    line_pred.set_ydata(y_pred)
    
    comp_length = min(end_act, end_pred)
    if comp_length > 0:
        actual_compare = y_actual[:comp_length]
        pred_compare = y_pred[:comp_length]
        mae_dynamic = mean_absolute_error(actual_compare, pred_compare)
        mse_dynamic = mean_squared_error(actual_compare, pred_compare)
        rmse_dynamic = np.sqrt(mse_dynamic)
        error_text.set_text(f'MAE: {mae_dynamic:.4f}, MSE: {mse_dynamic:.4f}, RMSE: {rmse_dynamic:.4f}')
    else:
        error_text.set_text('')
    
    # Segundo gráfico (ws100 original vs predictions)
    y_ws100_real = ws100_values[:end_act] if end_act > 0 else np.array([])
    y_pred2 = y_pred  # mesma y_pred do gráfico de cima
    
    line_actual2.set_xdata(time_actual)
    line_actual2.set_ydata(y_ws100_real)
    
    line_pred2.set_xdata(time_pred)
    line_pred2.set_ydata(y_pred2)
    
    if comp_length > 0 and len(y_ws100_real) == comp_length:
        actual_compare2 = y_ws100_real
        pred_compare2 = y_pred2[:comp_length]
        mae_dynamic2 = mean_absolute_error(actual_compare2, pred_compare2)
        mse_dynamic2 = mean_squared_error(actual_compare2, pred_compare2)
        rmse_dynamic2 = np.sqrt(mse_dynamic2)
        error_text2.set_text(f'MAE: {mae_dynamic2:.4f}, MSE: {mse_dynamic2:.4f}, RMSE: {rmse_dynamic2:.4f}')
    else:
        error_text2.set_text('')
    
    # Ajuste dos eixos
    ax.relim()
    ax.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    
    # Remove linhas verticais antigas nos dois eixos
    for axis in [ax, ax2]:
        # Filtra apenas as linhas pontilhadas pretas que criamos
        vertical_lines = [line for line in axis.lines if line.get_linestyle() == '--' and line.get_color() == 'black' 
                        and line is not line_actual and line is not line_pred 
                        and line is not line_actual2 and line is not line_pred2]
        for line in vertical_lines:
            line.remove()
    
    # Adiciona linha vertical pontilhada preta nos ticks principais de data
    for xtick in ax.get_xticks():
        ax.axvline(x=xtick, color='black', linestyle='--', linewidth=1, zorder=0)
    for xtick in ax2.get_xticks():
        ax2.axvline(x=xtick, color='black', linestyle='--', linewidth=1, zorder=0)
    
    plt.draw()

def advance(event):
    """Avançar na simulação"""
    global current_index
    current_index += step
    if current_index > len(actuals):
        current_index = len(actuals)
    update_plot()

def go_back(event):
    """Voltar na simulação"""
    global current_index
    current_index -= step
    if current_index < 0:
        current_index = 0
    update_plot()

# Adicionar botões de controle
ax_button_forward = plt.axes([0.65, 0.05, 0.05, 0.02])
button_forward = Button(ax_button_forward, 'Avançar')
button_forward.on_clicked(advance)

ax_button_back = plt.axes([0.35, 0.05, 0.05, 0.02])
button_back = Button(ax_button_back, 'Voltar')
button_back.on_clicked(go_back)

# Inicializar o gráfico
update_plot()

# Exibir a simulação
plt.show()
