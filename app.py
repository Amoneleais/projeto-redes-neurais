import pandas as pandas
import numpy as numpy
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from scipy.io import arff

# Numero de execuções pra rodar
n_execs = 30

# Configurações para testar
learning_rates = [0.1, 0.01, 0.001]
hidden_neurons = [3, 5, 7]

# Carregar dados do ARFF
raw = arff.loadarff('diabetes.arff')
data = pandas.DataFrame(raw[0])

# Mapear o rótulo 'class' para binário: positive = 1, negative = 0
data['class'] =data['class'].apply(lambda c: 1 if c == b'tested_positive' else 0)

# Definir features e rótulos
x_data = data.drop('class', axis=1)
y_data = data['class']

# Normalizar
sc = StandardScaler()
x_scaled = sc.fit_transform(x_data)

# Divisão dos dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_data, test_size=0.25, stratify=y_data, random_state=42)


# Função para adicionar matrizes de confusão
def sum_conf_matrizes(total_matrix, new_matrix):
    return total_matrix + new_matrix

# Loop nas taxas de aprendizado e neurônios
for lr in learning_rates:
    for neurons in hidden_neurons:
        
        # Inicializar acumuladores de MSE e matriz de confusão
        total_mse = 0
        conf_matrix_sum = numpy.zeros((2, 2)) # Inicializando para soma
        
        print("_"*65)
        print(f"\nTaxa de aprendizado: {lr} | Neurônios escondidos: {neurons}")

        # Executar várias vezes para calcular a média
        for exec_num in range(n_execs):
            
            # Definir o MLP (com essa taxa e número de neurônios)
            mlp = MLPClassifier(hidden_layer_sizes=(neurons,), learning_rate_init=lr, max_iter=1000, random_state=exec_num)
            mlp.fit(x_train, y_train)

            # Testar
            preds = mlp.predict(x_test)

            # Calcular MSE
            total_mse += mean_squared_error(y_test, preds)

            # Atualizar a matriz de confusão
            conf_matrix = confusion_matrix(y_test, preds)
            conf_matrix_sum = sum_conf_matrizes(conf_matrix_sum, conf_matrix)

        # Calcular e mostrar médias
        avg_mse = total_mse / n_execs
        avg_conf_matrix = conf_matrix_sum / n_execs

        print(f"\nMSE Médio após {n_execs} execuções.: {avg_mse:.4f}")
        print(f"Matriz de Confusão Média.:\n{avg_conf_matrix}")