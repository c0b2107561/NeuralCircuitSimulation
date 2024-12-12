import random
import numpy as np
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt
from generate_sine_wave_stripes import img_main


# Izhikevichモデルを用いたニューロンクラス
class IzhikevichNeuron:
    def __init__(self, excitatory=True, tau=1.0, decay_factor=0.85): # 0.1 1.0 0.5
        if excitatory: # 興奮性
            # self.a, self.b, self.c, self.d = 0.02, 0.2, -50, 2 # Excitatory neuron parameters CH
            self.a, self.b, self.c, self.d = 0.02, 0.2, -65, 8 # Excitatory neuron parameters RS
            # self.a, self.b, self.c, self.d = 0.02, 0.2, -55, 4 # Excitatory neuron parameters IB
        else: # 抑制性
            self.a, self.b, self.c, self.d = 0.1, 0.2, -65, 2  # Inhibitory neuron parameters FS
            # self.a, self.b, self.c, self.d = 0.02, 0.25, -65, 2  # Inhibitory neuron parameters LST

        # 初期状態
        self.v = self.c  # 初期膜電位
        self.u = self.b * self.v  # 回復変数
        self.last_spike_time = -float('inf')  # 最後のスパイク時間
        self.tau = tau
        self.decay_factor = decay_factor  # 忘却率

    def update(self, I, current_time, dt=0.3):
        # 膜電位を忘却率で減衰させる
        self.v *= self.decay_factor 

        if self.v >= 55:
            self.v = self.c
            self.u += self.d
            self.last_spike_time = current_time  # スパイク時刻を記録
            return 1.0  # スパイク発生
        
        # ノイズを膜電位に追加（ノイズの強さを調整可能）
        noise = np.random.normal(0, 1.5)  # 平均0, 標準偏差0.5のノイズ
        dv = 0.04 * self.v**2 + 5 * self.v + 140 - self.u + I + noise
        du = self.a * (self.b * self.v - self.u) 
        self.v += dv * dt
        self.u += du * dt
        return 0.0  # スパイクなし

# 入力層クラス
class InputLayer:
    def __init__(self, input_dim, n_reservoir_nodes):
        self.input_dim = input_dim  # 入力データ次元
        self.weights = np.random.randn(n_reservoir_nodes, input_dim)  # リザバー層への重み

    def process_input(self, input_data):
        """入力データをリザバー層へ変換"""
        noise = np.random.normal(0, 0.1, size=self.input_dim)  # 入力信号にノイズを追加
        return np.dot(self.weights, input_data + noise) 


# ニューロン集団クラス
class NeuronGroup:
    def __init__(self, n_neurons=1000, per=0.3, excitatory_ratio=0.8): # 2
        """ニューロン集団の初期化"""
        # 興奮性ニューロンと抑制性ニューロンの数を計算
        n_excitatory = int(n_neurons * excitatory_ratio)
        n_inhibitory = n_neurons - n_excitatory

        # 興奮性ニューロンと抑制性ニューロンをそれぞれ生成
        self.neurons = []
        self.neurons.extend([IzhikevichNeuron(excitatory=True) for _ in range(n_excitatory)])  # 興奮性ニューロン
        self.neurons.extend([IzhikevichNeuron(excitatory=False) for _ in range(n_inhibitory)])  # 抑制性ニューロン

        # ニューロンの位置をランダムに配置（2D空間）
        self.positions = np.random.rand(n_neurons, 3) * 500  # 各ニューロンの位置を[0, area_size]内でランダムに配置
        
        # ニューロンのランダム接続
        self.connections = self.random_connections(n_neurons, per)

        self.recorded_potentials = []  # 膜電位の記録

        self.spike_counts = [0] * n_neurons

        # ニューロン間の結合重みを初期化
        self.weights = {i: {j: np.random.uniform(0.1, 1.0) for j in connections}
                        for i, connections in self.connections.items()}

        # スパイクタイミングを記録する
        self.last_spike_times = [-float('inf')] * n_neurons


    def random_connections(self, n_neurons, per):
        """集団内のニューロン間のランダムな接続を生成"""
        
        connections = {i: [] for i in range(n_neurons)}  # 各ニューロンの接続リストを辞書で管理
        for i in range(n_neurons):
            for j in range(i + 1, n_neurons):
                if random.random() < per:  # 例えば30%の確率で接続
                    connections[i].append(j)  # iからjへの接続
                    connections[j].append(i)  # jからiへの接続
        return connections

    # def random_connections(self, n_neurons, per):
    #     """Watts-Strogatzモデルに基づくニューロン間の接続を生成"""
    #     # Watts-Strogatzグラフを生成
    #     k = int(per * n_neurons)  # 各ノードの近傍接続数
    #     # print(f"k : {k}")
    #     p_rewire = per  # 再接続確率（必要に応じて調整）
        
    #     # NetworkXでWatts-Strogatzグラフを生成
    #     # ws_graph = nx.watts_strogatz_graph(n_neurons, k, p_rewire)
    #     ws_graph = nx.watts_strogatz_graph(n_neurons, 6, p_rewire)

    #     # 接続を辞書形式に変換
    #     connections = {i: [] for i in range(n_neurons)}
    #     weights = {}

    #     for i, neighbors in ws_graph.adjacency():
    #         for neighbor in neighbors:
    #             weight = np.random.uniform(0.1, 1.0) if random.random() < p_rewire else np.random.uniform(0.5, 2.0)
    #             connections[i].append(neighbor)
    #             weights[(i, neighbor)] = weight

    #     self.connection_weights = weights
        
    #     return connections

    def stdp_update(self, pre_idx, post_idx, current_time, tau_pre=20.0, tau_post=20.0, A_plus=0.01, A_minus=0.012):
        """
        スパイクタイミング依存可塑性(STDP)に基づき結合重みを更新
        :param pre_idx: 発火したニューロンのインデックス（プリシナプス）
        :param post_idx: 発火したニューロンのインデックス（ポストシナプス）
        :param current_time: 現在の時間
        """
        delta_t = self.last_spike_times[post_idx] - self.last_spike_times[pre_idx]

        if delta_t > 0:
            delta_w = A_plus * np.exp(-delta_t / tau_pre)  # ポテンシエーション
        elif delta_t < 0:
            delta_w = -A_minus * np.exp(delta_t / tau_post)  # デプレッション
        else:
            return  # タイミングが一致しない場合、更新なし


    def update_group(self, I, current_time, dt=0.3): # 0.3896
        """集団内のニューロンを更新"""
        spikes = 0
        
        for idx, neuron in enumerate(self.neurons):
            spike = neuron.update(I, current_time, dt)
            spikes += spike

            if spike > 0:
                self.spike_counts[idx] += 1  # 発火回数を記録
                
                # 接続されているニューロンに対してSTDPを適用
                # for post_idx in self.connections[idx]:
                #     self.stdp_update(idx, post_idx, current_time)

        # 個々のニューロンの膜電位をそのまま返す
        return spikes, [neuron.v for neuron in self.neurons]
    
    def get_firing_frequencies(self, neuron_indices):
        """
        指定したニューロンの発火頻度を計算 (Hz単位)
        :param neuron_indices: 発火頻度を計算するニューロンのインデックスリスト
        :param total_time: シミュレーションの合計時間（秒）
        :return: ニューロンごとの発火頻度 (Hz)
        """
        frequencies = {idx: self.spike_counts[idx] for idx in neuron_indices}
        return frequencies

    def get_membrane_potentials(self):
        return [neuron.v for neuron in self.neurons]

    def record_potentials(self):
        self.recorded_potentials.append(self.get_membrane_potentials())
        # print(self.recorded_potentials)


    def plot_membrane_potential(self, neuron_idxs):
        """指定したニューロンの膜電位を複数プロット"""
        plt.figure(figsize=(10, 6))  # グラフのサイズを指定

        time_steps = np.arange(len(self.recorded_potentials))  # タイムステップの配列を生成

        
        # 指定された各ニューロンについて膜電位をプロット
        for neuron_idx in neuron_idxs:
            potentials = [self.recorded_potentials[step][neuron_idx] for step in range(len(self.recorded_potentials))]
            plt.plot(time_steps, potentials, label=f'Neuron {neuron_idx}')
        
        # グラフの設定
        plt.title('Membrane Potential of Selected Neurons')
        plt.xlabel('Time step')
        plt.ylabel('Membrane Potential (mV)')
        plt.legend()  # 凡例を表示
        plt.show()


# ニューロンネットワーククラス
class NeuronNetwork:
    def __init__(self, n_groups=100, n_neurons_per_group=1000, 
                 p_rewiring=0.1, spectral_radius=1.0, input_dim=10, per=0.3): # 1.15007
        """ネットワークの初期化"""
        # print(input_dim) # 1画像の色要素数(この値と同じ数だけinput層のノードが作成される)
        self.input_layer = InputLayer(input_dim, n_groups * n_neurons_per_group)  # 入力層の作成
        self.groups = [NeuronGroup(n_neurons_per_group, per) for _ in range(n_groups)]  # 集団を生成
        self.n_groups = n_groups
        self.p_rewiring = p_rewiring
        self.spectral_radius = spectral_radius
        self.rewire_connections()  # 集団間のWatts-Strogatzモデルによる接続の再接続
        self.reservoir_state = np.zeros((self.n_groups, n_neurons_per_group))  # リザバー層の状態を格納

        # グループの位置をランダムに配置（2D空間）
        self.positions = np.random.rand(n_groups, 3) * 50  # 各ニューロンの位置を[0, area_size]内でランダムに配置
        
    def rewire_connections(self):
        """集団間の結合をWatts-Strogatzモデルに基づき再接続"""
        G = nx.watts_strogatz_graph(self.n_groups, 6, self.p_rewiring)  # ノード数n_groups、隣接ノード数10、再接続確率p_rewiring
        adjacency_matrix = nx.to_numpy_array(G)
        max_eigenvalue = max(abs(np.linalg.eigvals(adjacency_matrix)))
        # self.reservoir_weights = adjacency_matrix / max_eigenvalue * self.spectral_radius 
        self.reservoir_weights = adjacency_matrix / max_eigenvalue * self.spectral_radius * 1.2 # 接続強化

    def update_network(self, input_data, current_time, dt=0.3, input_scale= 1.0): # 3.416
        """ネットワーク全体を更新"""
        spikes = 0
        avg_potentials = []  # 各集団の平均膜電位を格納
        I_scaled = self.input_layer.process_input(input_data) * input_scale  # 入力スケールを適用
        for i, group in enumerate(self.groups):
            group_spikes, group_avg_potential = group.update_group(I_scaled[i], current_time, dt)
            spikes += group_spikes
            avg_potentials.append(group_avg_potential)  # 集団ごとの平均膜電位を追加
            self.reservoir_state[i, :] = np.array([neuron.v for neuron in group.neurons])  # リザバー層の状態を更新

        self.reservoir_state = self.reservoir_state / np.linalg.norm(self.reservoir_state, axis=1, keepdims=True)  # 正規化

        
        # 平均膜電位を記録
        for i, group in enumerate(self.groups):
            group.recorded_potentials.append(avg_potentials[i])

        return spikes, avg_potentials  # スパイク数と各集団の平均膜電位を返す

    def get_reservoir_state(self):
        """リザバー層の状態を返す"""
        return self.reservoir_state.flatten()  # 平坦化して1D配列として返す

    @staticmethod
    def plot_recorded_potentials(groups, avg_potentials):
        plt.figure(figsize=(10, 6))
        
        # 複数のグループを指定して平均膜電位をプロット
        for group_idx in groups:
            if group_idx < len(avg_potentials):
                plt.plot(avg_potentials[group_idx], label=f'Group {group_idx}')
        
        plt.title('Average Membrane Potentials of Recorded Groups')
        plt.xlabel('Time step')
        plt.ylabel('Average Membrane Potential (mV)')
        plt.legend()
        plt.show()

# 出力層クラス
class OutputLayer:
    def __init__(self, input_dim, output_dim, learning_rate=0.1): # 0.3
        self.weights = np.random.randn(output_dim, input_dim) # 重みを小さな値で初期化
        self.biases = np.zeros(output_dim)  # バイアスを0で初期化
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        """シグモイド関数"""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """シグモイド関数の微分"""
        sigmoid_x = self.sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)

    def forward(self, reservoir_state):
        """順伝播計算"""
        self.reservoir_state = reservoir_state
        self.raw_output = np.dot(self.weights, reservoir_state) + self.biases  # 線形結合
        self.activated_output = self.sigmoid(self.raw_output)  # 活性化関数適用
        return self.activated_output

    def backward(self, output_error):
        """
        逆伝播計算: 
        - output_errorは出力層で計算された誤差 (e.g., 出力 - 正解)
        - 学習のための重みとバイアスの更新
        """
        raw_error = output_error * self.sigmoid_derivative(self.raw_output)  # 活性化関数の微分を適用
        weight_gradient = np.outer(raw_error, self.reservoir_state)  # 重みの勾配
        bias_gradient = raw_error  # バイアスの勾配

        # 重みとバイアスの更新
        self.weights -= self.learning_rate * weight_gradient
        self.biases -= self.learning_rate * bias_gradient

        # 次の層に伝える誤差
        return np.dot(self.weights.T, raw_error)

    def train(self, input_data, target_output):
        """入力と目標出力を用いて学習"""
        # 順伝播
        predicted_output = self.forward(input_data)

        # 誤差計算
        output_error = predicted_output - target_output

        # 逆伝播
        self.backward(output_error)

# 2-back課題における正答率の計算
def calculate_accuracy(predictions, correct_labels):
    correct_predictions = sum(np.round(pred) == label for pred, label in zip(predictions, correct_labels))
    accuracy = correct_predictions / len(correct_labels)
    return accuracy


def run_2back_task(data_list, network, output_layer, p_rewire):
    predictions = []  # 予測結果を格納
    correct_labels = [(1 if data_list[i] == data_list[i - 2] else 0) for i in range(2, len(data_list))]  # 正解ラベルを事前生成
    current_time = 0
    dt = 0.3  # 時間ステップ
    # total_time = 10  # 秒

    # 2-back課題のシミュレーション
    for step in range(2, len(data_list)):
        current_data = data_list[step]

        # ネットワークの更新
        spikes, avg_potentials = network.update_network(current_data, current_time, dt)
        reservoir_state = network.get_reservoir_state()

        # 出力層の学習
        output_layer.train(reservoir_state, correct_labels[step - 2])
        # output_layer.train([reservoir_state], [correct_labels[step - 2]])

        # 出力層の予測
        prediction = output_layer.forward(reservoir_state)
        # prediction = output_layer.predict(reservoir_state)
        predictions.append(int(prediction[0] > 0.5))  # 閾値で二値化

        current_time += dt  # 時間更新

    # グループ2のニューロン[0, 1, 2]の発火頻度を取得
    group_idxs = [0, 20, 40, 60, 80]
    neuron_indices = [0, 200, 400, 600, 800]
    for i in range(len(group_idxs)):
        firing_frequencies = network.groups[group_idxs[i]].get_firing_frequencies(neuron_indices)
        # print(f"グループ {group_idxs[i]} のニューロン発火頻度 (Hz): {firing_frequencies}")

    # グループ0の膜電位をプロット
    # network.groups[60].plot_membrane_potential(neuron_idxs=neuron_indices)

    # # グループ0，10，20の平均膜電位をプロット
    # # gs = [0, 10, 20]
    # # network.plot_recorded_potentials(gs, avg_potentials)
    
    # 正答率を計算
    accuracy = calculate_accuracy(predictions, correct_labels)
    accuracy = round(accuracy, 2)
    print(f"p_rewire {p_rewire} : 2-back Task Accuracy: {accuracy}")
    return accuracy


def main(num):
    # per_list = [0.007, 0.07, 0.7] # WS
    per_list = [0.3, 0.3, 0.3] # [0.05, 0.3, 0.5] random
    
    p_rewire_list = [0.007, 0.07, 0.7] # 0.7
    ac_lists = []
    for i in range(len(p_rewire_list)):
        ac_list = []
        for j in range(num):
            # データセット
            data_list = img_main()
            
            
            # ニューロンネットワークの作成
            network = NeuronNetwork(n_groups=100, n_neurons_per_group=1000, p_rewiring=p_rewire_list[i]
                                    , input_dim=len(data_list[0]), per=per_list[i])

            # 出力層の作成
            output_layer = OutputLayer(input_dim=network.n_groups * 1000, output_dim=len(data_list[0]))

            # 2-back課題の実行
            accuracy = run_2back_task(data_list, network, output_layer, p_rewire_list[i])
            ac_list.append(accuracy)
        ac_lists.append(ac_list)
    print(ac_lists)
    return ac_lists

if __name__ == "__main__":
    num = 3
    main(num)
