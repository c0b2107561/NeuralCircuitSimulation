import numpy as np
import matplotlib.pyplot as plt
import network_img as net
import networkx as nx
import time
import csv
from generate_sine_wave_stripes import main

p_rewire_list = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

class SpikingEchoStateNetwork:
    def __init__(self, n_inputs, n_reservoir, n_outputs, spectral_radius=1, tau=0.03, threshold=1.0, reset_value=0.0, k_neighbors=6, p_rewire=0.3):
        self.n_inputs = n_inputs 
        self.n_reservoir = n_reservoir 
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.tau = tau  # 膜電位の減衰のタイム定数
        self.threshold = threshold  # 発火閾値
        self.reset_value = reset_value  # 発火後の膜電位のリセット値

        # Watts-Strogatzモデルによるスモールワールドネットワークの生成
        ws_graph = net.Watts_Strogats_small_world_graph(n_reservoir, k_neighbors, p_rewire)

        # 隣接行列を取得
        adj_matrix = nx.to_numpy_array(ws_graph)

        # リザーバの重み行列を生成し、スペクトル半径を調整
        self.W_reservoir = adj_matrix * (np.random.rand(n_reservoir, n_reservoir) - 0.5)
        rhoW = max(abs(np.linalg.eigvals(self.W_reservoir)))
        self.W_reservoir *= self.spectral_radius / rhoW

        # 入力の重みを初期化
        self.W_input = np.random.rand(n_reservoir, n_inputs) - 0.5

        # 出力の重みを初期化
        self.W_output = np.zeros((n_outputs, n_reservoir))

        # リザーバの状態を初期化
        self.membrane_potential = np.zeros(n_reservoir)
        self.spikes = np.zeros(n_reservoir)

    def update(self, input_data, dt=1.0):
        # 膜電位を更新
        self.membrane_potential += (-self.membrane_potential + np.dot(self.W_input, input_data)) / self.tau * dt
        
        # 発火のチェック
        self.spikes = (self.membrane_potential > self.threshold).astype(float)
        
        # 発火後に膜電位をリセット
        self.membrane_potential[self.spikes > 0] = self.reset_value
        
        return self.spikes

    def train(self, inputs, targets, reg=1e-6):
        states = np.zeros((len(inputs), self.n_reservoir))
        for t in range(len(inputs)):
            states[t] = self.update(inputs[t])
        
        # 出力の重みを計算
        self.W_output = np.dot(np.dot(targets.T, states), np.linalg.inv(np.dot(states.T, states) + reg * np.eye(self.n_reservoir)))

    def predict(self, input_data):
        spikes = self.update(input_data)
        return np.dot(self.W_output, spikes)

def generate_n_back_data(sequences, n):
    inputs = []
    targets = []
    for i in range(len(sequences) - n):
        inputs.append(sequences[i])
        targets.append(sequences[i + n])
    return np.array(inputs), np.array(targets)

if __name__ == "__main__":
    sequences = main()
    
    n_back = 2
    n_inputs = len(sequences[0])
    n_reservoir = 100
    n_outputs = len(sequences[0])

    # データ生成
    inputs, targets = generate_n_back_data(sequences, n_back)

    # スパイキングエコーステートネットワークのインスタンス化
    esn = SpikingEchoStateNetwork(n_inputs, n_reservoir, n_outputs)

    # モデルのトレーニング
    esn.train(inputs, targets)

    # 予測の実行と結果表示
    predictions = np.array([esn.predict(input_data) for input_data in inputs])

    # 予測結果をバイナリに変換
    binary_predictions = (predictions > 0.5).astype(int)

    # 結果の表示
    results = []
    for i, (input_data, binary_prediction) in enumerate(zip(inputs, binary_predictions)):
        print(f"入力: {input_data}, 予想出力: {binary_prediction}, 正しい出力: {targets[i]}")
        results.append([input_data, binary_prediction, targets[i]])
        time.sleep(0.03)

    # 精度の計算
    accuracy = np.mean(np.all(binary_predictions == targets, axis=1))
    print(f"精度:{accuracy}")
    results.append(accuracy)
    
    # 結果のプロット
    # plt.figure(figsize=(12, 6))
    # for i in range(n_outputs):
    #     plt.plot(targets[:, i], label=f'correct_output (Targets) {i+1}')
    #     plt.plot(predictions[:, i], label=f'real_autput (Predictions) {i+1}')
    # plt.legend()
    # plt.title(f'n-back Task Predictions\naccuracy: {accuracy:.4f}')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Values')
    # plt.show()
