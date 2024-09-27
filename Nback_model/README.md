# Echo State Network n-back model

## 使用言語
<img src="https://img.shields.io/badge/Python-3.12.1-3776AB.svg?logo=python&style=flat-square"> 

## 目次  
1. [概要](#概要)
2. [各プログラムについて](#各プログラムについて)
3. [echo_state_network.py](echo_state_network.py)
4. [generate_sine_wave_stripes.py](generate_sine_wave_stripes.py)
5. [network_img.py](network_img.py)

## 概要
このフォルダはスモールワールド性を有するエコーステートネットワークを作成し，中間層の構造を変化させて健常者とASDに相当するモデルとする．各モデルにWM機能を評価することが可能なN-Back課題を遂行させ，その精度によりネットワーク構造がWM機能に与える影響を分散分析と多重比較により評価する．

## 各プログラムについて
1. echo_state_network.py
　最も基本的なプログラム．
2. echo_state_network_v2.py
　学習則をスパイクタイミング依存可塑性(STDP)としたプログラム．
3. echo_state_network_v3.py
　リザバー層を入力付近と出力付近と入出間の3つに分けたプログラム(試作中)．
4. echo_state_network_v4.py
　フィードバックをリストとして保持するようにしたプログラム．
5. echo_state_network_v5.py
　3を基に実験的に値を変化させたり試行するためのプログラム．

## echo_state_network.py

```python:echo_state_network.py  
```

## generate_sine_wave_stripes.py

## network_img.py
### 機能
Watts-Strogatsモデルを用いたリザバー層の構築．

### ライブラリ
### 変数
