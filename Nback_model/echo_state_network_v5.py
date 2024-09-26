import numpy as np
import matplotlib.pyplot as plt
import network_img as net
import networkx as nx
import time
import csv
import pandas as pd
from generate_sine_wave_stripes import img_main


class SpikingEchoStateNetwork:
    # def __init__(self, n_inputs, n_reservoir, n_outputs, spectral_radius=1.0, tau=0.02, threshold=1.0, reset_value=0.0, k_neighbors=6, p_rewire=0.3, A_plus=0.01, A_minus=0.01, tau_plus=20.0, tau_minus=20.0): # 既存
    def __init__(self, n_inputs, n_reservoir, n_outputs, p_rewire,
                 spectral_radius=1.0, tau=0.01, threshold=1.0, 
                 reset_value=0.0, k_neighbors=6, A_plus=0.5, 
                 A_minus=0.5, tau_plus=50.0, tau_minus=50.0): # 最適化 
        
        self.n_inputs = n_inputs # input層のニューロン数, 1N:1要素
        self.n_reservoir = n_reservoir # reserver層のニューロン数 
        self.n_outputs = n_outputs # output層のニューロン数
        self.p_rewire = p_rewire # 再配線確率．低:近傍接続数多
        self.spectral_radius = spectral_radius # reserver層の重み行列のスペクトル半径
        self.tau = tau # ニューロンの膜電位の原則速度を決定する時間定数[大きいと減衰が遅くなる]
        self.threshold = threshold # ニューロンがスパイクを発生されるための閾値
        self.reset_value = reset_value # スパイクを発生させたニューロンの膜電位をリセットする値

        ## STDP parameters
        self.A_plus = A_plus # 発火後のシナプス強化量を決定する
        self.A_minus = A_minus # 発火後のシナプス減衰量を決定する
        self.tau_plus = tau_plus # 発火後のシナプス強化がどれの時間まで影響するかを決める時間定数
        self.tau_minus = tau_minus # 発火後のシナプス減衰がどれの時間まで影響するかを決める時間定数


        # self.input_region_size = n_reservoir // 3 # input層に対応するreserver層の領域サイズ
        self.input_region_size = 80
        # self.output_region_size = n_reservoir // 3 # output層に対応するreserver層の領域サイズ
        self.output_region_size = 80
        self.hidden_region_size = n_reservoir - self.input_region_size - self.output_region_size

        ## Small-world network creation (reserver層の初期化)
        # スモールワールドネットワーク作成
        ws_graph = net.Watts_Strogats_small_world_graph(n_reservoir, k_neighbors, p_rewire)
        # ws_graphを隣接行列に変換(ノードの接続関係を示す[接続有:1])
        adj_matrix = nx.to_numpy_array(ws_graph) 
        # 重み行列[ランダム値を-0.5から0.5範囲にシフト]
        self.W_reservoir = adj_matrix * (np.random.rand(n_reservoir, n_reservoir) - 0.5) 
        # 現スペクトル半径[np.linalg.evals:W_reserverの固有値を計算]
        rhoW = max(abs(np.linalg.eigvals(self.W_reservoir)))
        # 重み行列全体にスケーリングを行い，スペクトル半径を1に近づける[ネットワークダイナミクスの制御]
        self.W_reservoir *= self.spectral_radius / rhoW

        # 重み行列を初期化[接続:input → reserver input region]
        self.W_input = np.random.rand(self.input_region_size, n_inputs) - 0.5 
        # 初期の重み行列を0に初期化, 学習プロセスで調整される[reserver output region → output]
        self.W_output = np.zeros((n_outputs, self.output_region_size))
        # reserver層のニューロン膜電位を0で初期化．閾値到達でスパイク発生
        self.membrane_potential = np.zeros(n_reservoir)
        # 各ニューロンがスパイクを発生させたかを示す状態を0で初期化
        self.spikes = np.zeros(n_reservoir)
        # 各ニューロンが最後にスパイクを発生させた時間を保持．初期は無限に長い時間[-np.inf]
        self.last_spike_times = np.full(n_reservoir, -np.inf)

    def update(self, input_data, current_time, dt=1):
        # ネットワークの状態更新とスパイク生成
        ## input region
        # reserver層のinput領域の膜電位を取得
        input_region_potential = self.membrane_potential[:self.input_region_size]
        # 膜電位の変化率を計算し,dt(時間刻み)で更新する[np.dot():入力信号をreserver層のinput領域に追加]
        input_region_potential += (-input_region_potential + np.dot(self.W_input, input_data)) / self.tau * dt
        # 膜電位が閾値を超えた場合にスパイク発生とみなす[浮動小数に変換(スパイクの有無 0 or 1)]
        self.spikes[:self.input_region_size] = (input_region_potential > self.threshold).astype(float)

        ## hidden and output region
        # self.membrane_potential[]:隠れ領域および出力領域の膜電位を取得
        # np.dot():スパイク状態とreserver層の重み行列のドット積を計算し，膜電位を更新
        self.membrane_potential[self.input_region_size:] += (-self.membrane_potential[self.input_region_size:] +
                                                             np.dot(self.W_reservoir[self.input_region_size:], self.spikes)) / self.tau * dt
        # 膜電位が閾値を超えた場合にスパイク発生とみなす[浮動小数に変換(スパイクの有無 0 or 1)]
        self.spikes[self.input_region_size:] = (self.membrane_potential[self.input_region_size:] > self.threshold).astype(float)

        count = 0

        ## Apply STDP
        for i in range(self.n_reservoir): # 100
            if self.spikes[i] > 0:
                # 各ニューロンの最後のスパイクからの時間差を計算する
                delta_t = current_time - self.last_spike_times
                # スパイクが発生したニューロンの重みを増加させる
                self.W_reservoir[:, i] += self.A_plus * np.exp(-delta_t / self.tau_plus)
                # スパイクが発生したニューロンの重みを減少させる
                self.W_reservoir[i, :] -= self.A_minus * np.exp(-delta_t / self.tau_minus)
                # スパイク発生時刻を更新します
                self.last_spike_times[i] = current_time

                # 重み0って接続されてないよね
                # for j in range(len(self.W_reservoir[:, i])):
                #     reservoir_list = self.W_reservoir[:, i]
        #             if reservoir_list[j] == 0:
        #                 count += 1
        #                 print(f"リザバー層の重み{reservoir_list[j]}")
        # print(len(self.W_reservoir)) # 100 
        # print(len(reservoir_list)) # 100 (1ノードから他全てのノードへの接続重み)
        # print(f"重み0のcount:{count}") # p=0.3/67, p=0/70

        
        ## Reset membrane potentials
        # スパイクが発生したニューロンの膜電位をリセットする
        self.membrane_potential[self.spikes > 0] = self.reset_value
        
        return self.spikes

    
    ## トレーニングデータを使ってネットワークを訓練する 
    def train(self, inputs, targets, reg=1e-6):
        # 入力データに対してupdateメゾットを呼び出し,reserver層の状態を収集する
        states = np.zeros((len(inputs), self.n_reservoir))
        for t in range(len(inputs)):
            states[t] = self.update(inputs[t], t)
        
        ## Calculate output weights
        # reserver層の出力領域の状態を取り出す
        output_states = states[:, -self.output_region_size:]
        # np.dot(targets.T, output_states):ターゲットデータとreserver層の出力状態のドット積を計算
        # np.dot()+reg * np.eye():reserver層の出力状態の共分散行列を計算し,リッジ回帰(正規化)用の項を追加する
        self.W_output = np.dot(np.dot(targets.T, output_states), np.linalg.inv(np.dot(output_states.T, output_states) + reg * np.eye(self.output_region_size)))
    
    ## 訓練済みのネットワークを使って予測を行う
    def predict(self, input_data):
        # スパイクの更新(入力データに対するスパイク状態の取得)
        spikes = self.update(input_data, 0)  # この時点でSTDPは適応されない
        output_spikes = spikes[-self.output_region_size:] # reserver層の出力領域のスパイク状態を取り出す
        return np.dot(self.W_output, output_spikes) # 出力重みと出力スパイク上体のドット積を計算し，最終的な予測値を得る

# n-backタスク用のデータセットを生成
def generate_n_back_data(sequences, n1, n2): 
    inputs = []
    targets = []
    for i in range(len(sequences) - n1):
        inputs.append(sequences[i])
        targets.append(sequences[i + n1])
    return np.array(inputs), np.array(targets)



def main():
    p_rewire_list = [0.0, 0.002, 0.005, 0.01, 0.02, 0.35, 0.05, 
                     0.1, 0.13, 0.17, 0.2, 0.23, 0.27,  0.3, 
                     0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # p_rewire_list = [0.0, 0.15, 0.3, 0.4, 0.9, 1.0]
    # p_rewire_list = [0.0, 0.3, 1.0]

    rep = 30
    
    ac_lists = []
    for i in range(len(p_rewire_list)):
        sequences = img_main()
        n_back = 2
        n_inputs = len(sequences[0])  # 40
        n_reservoir = 170
        # n_reservoir = 100
        n_outputs = len(sequences[0])  # 40

        # for i in range(len(p_rewire_list)):
        print(len(p_rewire_list))
        p_rewire = p_rewire_list[i]
        print(p_rewire)

        # ---------- # In this case, there are two types of list elements. Why?
        # # データ生成
        # inputs, targets = generate_n_back_data(sequences, n_back)

        # # スパイキングエコーステートネットワークのインスタンス化
        # esn = SpikingEchoStateNetwork(n_inputs, n_reservoir, n_outputs, p_rewire)

        # # モデルのトレーニング
        # esn.train(inputs, targets)

        # ac_list = []
        # for i in range(rep):
        # ---------- #

        # ~~~~~~~~~~ # In this case, there are many kinds of list element.
        ac_list = []
        for j in range(rep):
            for k in range(len(sequences)):
                inputs, targets = generate_n_back_data(sequences[k], n_back, len(sequences))
                esn = SpikingEchoStateNetwork(n_inputs, n_reservoir, n_outputs, p_rewire)

                esn.train(inputs, targets)
            # ~~~~~~~~~~ #
            
                predictions = np.array([esn.predict(input_data) for input_data in inputs])

                binary_predictions = (predictions > 0.5).astype(int)

                results = []
                for i, (input_data, binary_prediction) in enumerate(zip(inputs, binary_predictions)):
                    # print(f"入力: {input_data}, 予想出力: {binary_prediction}, 正しい出力: {targets[i]}")
                    results.append([input_data, binary_prediction, targets[i]])
                    # time.sleep(0.03)
                    time.sleep(5)

                accuracy = np.mean(np.all(binary_predictions == targets, axis=1))
                ac = '{:.3f}'.format(accuracy)
                # print(f"精度:{ac}")
                ac_list.append(ac)
                # print(ac_list)
        ac_lists.append(ac_list)
        # print(ac_lists)

    Columns = []
    for i in range(rep):
        Columns.append(i+1)
    # print(Columns)

    # df = pd.DataFrame(ac_lists, columns = Columns, index = p_rewire_list)
    # # df = df.transpose()
    # df.to_csv("v4/data/analysis_stdp_v3.csv")
    
    
    # リストCSV化
    # with open("v4/data/analysis_stdp.csv", "a") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(ac_lists)
    
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

    return ac_lists

    # 先生に聞く：STDPでは重みの更新によって不要な接続が除去される
    # 不要な接続除去されたら()それ刈込不全じゃなくね？笑

if __name__ == "__main__":
    main()
