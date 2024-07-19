import numpy as np
import networkx as nx
import network_img as net
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
import pandas as pd

class EchoStateNetwork:
    def __init__(self, n_inputs, n_reservoir, n_outputs, spectral_radius=0.6, sparsity=0.1): 
        # 初期化関数
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity

        # Watts-Strogatzモデルによるスモールワールドネットワークの生成
        ws_graph = net.Watts_Strogats_small_world_graph()

        # 隣接行列を取得
        adj_matrix = nx.to_numpy_array(ws_graph)

        # リザーバの重み行列を生成し、スペクトル半径を調整
        self.W_reservoir = adj_matrix * (np.random.rand(n_reservoir, n_reservoir) - 0.5)
        rhoW = max(abs(np.linalg.eigvals(self.W_reservoir)))
        self.W_reservoir *= self.spectral_radius / rhoW

        # 入力の重み行列をランダムに生成
        self.W_input = np.random.rand(n_reservoir, n_inputs) - 0.5

        # 出力の重み行列を初期化
        self.W_output = np.zeros((n_outputs, n_reservoir))

        # リザーバの状態を初期化
        self.state = np.zeros(n_reservoir)

    def update(self, input_data):
        # リザーバの状態を更新
        self.state = np.tanh(np.dot(self.W_input, input_data) + np.dot(self.W_reservoir, self.state))
        return self.state

    def train(self, inputs, targets, reg=1e-6):
        # リザーバ状態の記録
        states = np.zeros((len(inputs), self.n_reservoir))
        for t in range(len(inputs)):
            states[t] = self.update(inputs[t])

        # 正則化付きリッジ回帰で出力重みを学習
        self.W_output = np.dot(np.dot(targets.T, states), np.linalg.inv(np.dot(states.T, states) + reg * np.eye(self.n_reservoir)))

    def predict(self, input_data):
        # 入力データに対する予測を実行
        state = self.update(input_data)
        return np.dot(self.W_output, state)

def generate_n_back_data(sequence, n):
    # n-back課題のデータ生成
    inputs = np.zeros((len(sequence) - n, len(set(sequence))))
    targets = np.zeros((len(sequence) - n, len(set(sequence))))
    for i in range(len(sequence) - n):
        inputs[i, sequence[i]] = 1
        targets[i, sequence[i + n]] = 1
    return inputs, targets

def hyperparameter_optimization(param_grid, n_inputs, n_outputs, n_back, sequence_length, test_sequence_length):
    best_score = 0
    best_params = None

    results = []

    for params in ParameterGrid(param_grid):
        n_reservoir = params['n_reservoir']
        spectral_radius = params['spectral_radius']
        sparsity = params['sparsity']
        reg = params['reg']

        sequence = np.random.randint(0, n_inputs, sequence_length)
        inputs, targets = generate_n_back_data(sequence, n_back)

        esn = EchoStateNetwork(n_inputs, n_reservoir, n_outputs, spectral_radius, sparsity)
        esn.train(inputs, targets, reg)

        test_sequence = np.random.randint(0, n_inputs, test_sequence_length)
        test_inputs, test_targets = generate_n_back_data(test_sequence, n_back)
        predictions = [esn.predict(test_input) for test_input in test_inputs]

        predicted_labels = np.argmax(predictions, axis=1)
        actual_labels = np.argmax(test_targets, axis=1)
        accuracy = accuracy_score(actual_labels, predicted_labels)

        results.append({
            'n_reservoir': n_reservoir,
            'spectral_radius': spectral_radius,
            'sparsity': sparsity,
            'reg': reg,
            'accuracy': accuracy
        })

        if accuracy > best_score:
            best_score = accuracy
            best_params = params

    return best_params, best_score, results

# ハイパーパラメータのグリッドを定義
param_grid = {
    #'n_reservoir': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'n_reservoir': [100],
    'spectral_radius': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'sparsity': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'reg': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
}

# パラメータの設定
n_inputs = 3
n_outputs = 3
n_back = 2
sequence_length = 100
test_sequence_length = 100

# ハイパーパラメータの最適化を繰り返し実行
num_iterations = 10
all_results = []

for i in range(num_iterations):
    best_params, best_score, results = hyperparameter_optimization(param_grid, n_inputs, n_outputs, n_back, sequence_length, test_sequence_length)
    all_results.extend(results)
    print(f"Iteration {i+1}/{num_iterations} - Best Score: {best_score} - Best Params: {best_params}")

# データフレームに結果を格納
df = pd.DataFrame(all_results)

# 結果をExcelファイルに保存
excel_filename = "esn_hyperparameter_optimization_results.xlsx"
df.to_excel(excel_filename, index=False)
print(f"Results saved to {excel_filename}")
