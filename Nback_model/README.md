# Echo State Network n-back model

## 使用言語
<img src="https://img.shields.io/badge/Python-3.12.1-3776AB.svg?logo=python&style=flat-square"> 

## 目次  
1. [概要](#概要)
2. [ネットワークイメージ](#ネットワークイメージ)
3. [各プログラムについて](#各プログラムについて)
4. [echo_state_network.py](echo_state_network.py)
5. [generate_sine_wave_stripes.py](generate_sine_wave_stripes.py)
6. [mulyiple_comperisons](mulyiple_comparisons.py)
7. [network_img.py](network_img.py)
8. [vaiance_analysis.py](variance_analysis.py)

## 概要
このフォルダに存在するプログラムは作成した複数のエコーステートネットワークにN-Back課題を遂行させた時の正答率を分散分析と多重比較により評価してネットワーク構造がWM機能に与える影響を評価することを目的として作成した．

## ネットワークイメージ

## 各プログラムについて
1. echo_state_network.py  
ネットワークの作成と2-back課題の遂行，正答率の算出と保存を行う
2. generate_sine_wave_stripes.py  
2-back課題に使用する正弦波縞画像を複数作成する
3. mulyiple_comparisons.py  
Tukey法とFisher法による多重比較を行う
4. network_img.py  
Watts-Strogats modelによって作成されるネットワーク構造のイメージ
5. variance_analysis.py  
F検定による分散分析を行う

## echo_state_network.py
ネットワークの作成と2-back課題の遂行，正答率の算出と保存を行う．

### IzhikevichNeuronクラス
Izhikevichモデルを用いてニューロンの初期化と膜電位の変化や発火判定を行う．
``` python
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
```
- ニューロンの種類  
  - 興奮性ニューロン：chattering(CH)，regular spiking(RS)，intrinsically bursting(IB)  
  - 抑制性ニューロン：fast spiking(FS)，low-threshold spiking(LTS)  
- 膜電位の更新式
  - 0.04 * 膜電位^2 + 5 * 膜電位 + 140 - 回復変数 + 入力電流 + ノイズ
  - 回復変数：ニューロンの過剰活性を防ぐ
  - ノイズ：膜電位に加える．ニューロン活動の不確実性を表現してよりリアルな挙動を再現

### InputLayerクラス
リザバー層にデータを入力するための入力層を作成，リザバー層が扱う初期データを作成する
``` python
# 入力層クラス
class InputLayer:
    def __init__(self, input_dim, n_reservoir_nodes):
        self.input_dim = input_dim  # 入力データ次元
        self.weights = np.random.randn(n_reservoir_nodes, input_dim)  # リザバー層への重み

    def process_input(self, input_data):
        """入力データをリザバー層へ変換"""
        noise = np.random.normal(0, 0.1, size=self.input_dim)  # 入力信号にノイズを追加
        return np.dot(self.weights, input_data + noise) 
```
- 入力層のノード数 = input_dim
- self.weights：入力層とリザバー層の間の結合強度の行列をランダムで初期化
- ノイズ：入力データに加える．リザバー層の動きをより多様にする
- リザバー層への入力値：input_data+noiseとself.weightsの行列積

### NeuronGroupクラス
各集団内に存在する1000個のニューロンをシミュレーションする，発火や接続や膜電位の描画を行う
``` python
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
```
- ニューロン位置：三次元空間で各軸の座標が[0, 500]の範囲にランダムで設定(ある程度存在範囲を固定することが可能になる)
- self.weights：ニューロン間の結合強度を設定（0.1 - 1.0）
- 各ニューロンに対してランダムに他ニューロンとの接続を作る（確率30%）
