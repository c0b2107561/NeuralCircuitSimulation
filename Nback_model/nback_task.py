import numpy as np
import networkx as nx
import network_img as net

class EchoStateNetwork:
    def __init__(self, n_inputs, n_reservoir, n_outputs, spectral_radius=0.6, sparsity=0.1): # spectral_radius=1.25
        # 初期化関数
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius

        # Watts-Strogatzモデルによるスモールワールドネットワークの生成
        ws_graph = net.Watts_Strogats_small_world_graph()

        # 隣接行列を取得
        adj_matrix = nx.to_numpy_array(ws_graph)

        # リザーバの重み行列をランダムに生成し、スペクトル半径を調整
        # self.W_reservoir = np.random.rand(n_reservoir, n_reservoir) - 0.5
        # self.W_reservoir[np.random.rand(*self.W_reservoir.shape) > sparsity] = 0
        # radius = np.max(np.abs(np.linalg.eigvals(self.W_reservoir)))
        # self.W_reservoir *= spectral_radius / radius

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

    def train(self, inputs, targets, reg=1e-6): #reg=1e-8
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

# パラメータの設定
n_inputs = 3
n_reservoir = 100 #50 #ノード数100のため，100が限界値
n_outputs = 3
n_back = 2

# シーケンスデータの生成
sequence = np.random.randint(0, n_inputs, 100)
inputs, targets = generate_n_back_data(sequence, n_back)

# ESNの初期化
esn = EchoStateNetwork(n_inputs, n_reservoir, n_outputs)

# トレーニング
esn.train(inputs, targets)

# テスト
test_sequence = np.random.randint(0, n_inputs, 100) # 20
#test_sequence = np.random.randint(0, n_inputs, 10) # 2進数リスト確認用
test_inputs, test_targets = generate_n_back_data(test_sequence, n_back)
predictions = [esn.predict(test_input) for test_input in test_inputs]

# 予測と実際の出力を比較して精度を計算
predicted_labels = np.argmax(predictions, axis=1)
actual_labels = np.argmax(test_targets, axis=1)
accuracy = np.mean(predicted_labels == actual_labels)

# 結果の表示
# print("入力:", test_inputs) # 2進数リスト
# print("ターゲット:", test_targets) # 2進数リスト
# print("テスト入力:", test_sequence[n_back:])
# print("予測出力:", predicted_labels)
# print("実際の出力:", actual_labels)
print("精度:", accuracy)
