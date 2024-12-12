# Echo State Network n-back model

## 使用言語
<img src="https://img.shields.io/badge/Python-3.12.1-3776AB.svg?logo=python&style=flat-square"> 

## 目次  
1. [概要](#概要)
2. [各プログラムについて](#各プログラムについて)
3. [echo_state_network.py](echo_state_network.py)
4. [generate_sine_wave_stripes.py](generate_sine_wave_stripes.py)
5. [mulyiple_comperisons](mulyiple_comparisons.py)
6. [network_img.py](network_img.py)
7. [vaiance_analysis.py](variance_analysis.py)

## 概要
このフォルダに存在するプログラムは作成した複数のエコーステートネットワークにN-Back課題を遂行させた時の正答率を分散分析と多重比較により評価してネットワーク構造がWM機能に与える影響を評価することを目的として作成した．

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
