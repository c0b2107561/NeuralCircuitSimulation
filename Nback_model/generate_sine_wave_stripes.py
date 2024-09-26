import numpy as np
import matplotlib.pyplot as plt
import random

# 正弦波縞画像を生成する関数
def generate_sine_wave_stripes(width, height, frequency, amplitude):
    # x軸とy軸の値を生成
    x = np.linspace(0, 2 * np.pi * frequency, width)
    y = np.linspace(0, 2 * np.pi * frequency, height)
    # メッシュグリッドを作成
    X, Y = np.meshgrid(x, y)
    # 正弦波パターンを生成
    Z = amplitude * np.sin(X)
    return Z

# 正弦波縞画像をプロットする関数
def plot_sine_wave_stripes(Z):
    plt.figure(figsize=(6, 5))  # プロットのサイズを指定
    plt.imshow(Z, cmap='gray', origin='lower', aspect='auto')  # 画像をプロット
    plt.colorbar()  # カラーバーを表示
    plt.title('Sine Wave Stripes')  # タイトルを設定
    plt.xlabel('X-axis')  # x軸のラベルを設定
    plt.ylabel('Y-axis')  # y軸のラベルを設定
    plt.show()  # プロットを表示

# 各領域の色を判断する関数
def determine_region_colors(Z, num_regions_x, num_regions_y):
    height, width = Z.shape  # 画像の高さと幅を取得
    region_height = height // num_regions_y  # 各領域の高さを計算
    region_width = width // num_regions_x  # 各領域の幅を計算
    region_colors = []  # 各領域の色を格納するリスト

    # 各領域をループして色を判断
    for i in range(num_regions_y):
        row_colors = []  # 各行の色を格納するリスト
        for j in range(num_regions_x):
            # 領域を切り出し
            region = Z[i*region_height:(i+1)*region_height, j*region_width:(j+1)*region_width]
            # 領域の平均色を計算
            mean_color = np.mean(region)
            # 平均色が正なら1（白）、負なら0（黒）をリストに追加
            if mean_color > 0:
                row_colors.append(1)
            else:
                row_colors.append(0)
        region_colors.append(row_colors)  # 行の色を2重リストに追加

    return region_colors

def generate_and_analyze_sine_wave(width, height, amplitude, num_regions_x, num_regions_y, frequency):

    # 正弦波縞を生成
    Z = generate_sine_wave_stripes(width, height, frequency, amplitude)

    # 正弦波縞をプロット
    # plot_sine_wave_stripes(Z)

    # 各領域の色を判断
    region_colors = determine_region_colors(Z, num_regions_x, num_regions_y)
    return region_colors

def img_main():
    width = 400
    height = 400
    amplitude = 1
    num_regions_x = 40 #これで各周波数のリストに差が出る
    num_regions_y = 40

    # frequencies = [1, 3, 5, 7, 9, 11]  # 周波数の要素一覧
    # frequencies = [1, 3, 5, 9, 5, 1, 11, 7, 11, 3, 9, 3, 7, 1, 7, 5, 5, 1, 11, 1, 9] # test_data # 21
    frequencies = [1, 3, 5, 9, 5, 1, 11, 7, 11, 3, 9, 3, 7, 1, 7, 5, 5, 1, 11, 1, 
                   9, 3, 9, 5, 7, 5, 1, 11, 3, 11, 3, 9, 1, 9, 7, 11, 5, 3, 5, 5, 1] # test_data # 40 # 1を足して要素数41が元
    # frequencies = [1, 9, 5, 3, 5] # test_data
    print(len(frequencies))
    data_list = []
    
    #random
    # frequency = random.choice(frequencies) # 周波数をランダムに選択
    # print(frequency)
    # region_colors = generate_and_analyze_sine_wave(width, height, amplitude, num_regions_x, num_regions_y, frequency)
    # print(f"Region colors:{region_colors[1]}")
    
    # not random
    for i in range(len(frequencies)):
        # print(frequencies[i])
        region_colors = generate_and_analyze_sine_wave(width, height, amplitude, num_regions_x, num_regions_y, frequencies[i])
        # print(f"Region colors:{region_colors[1]}")
        data_list.append(region_colors[1])
    # print(data_list)
    # print(len(data_list[0]))
    return data_list

if __name__ == "__main__":
    img_main()
