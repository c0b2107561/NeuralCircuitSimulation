# HH Model Program

## 実行方法

1, ディレクトリ構成．  
<p align="center">
<img src="https://github.com/c0b2107561/NeuralCircuitSimulation/HHmodel/blob/main/tree.png" width="350px">
</p>

・**実行ファイルがあるディレクトリ**にいる状態で下記実行．  

### コンパイル

``` powershell
gcc -O3 -std=gnu11 -Wall -c hh.c
gcc -O3 -std=gnu11 -Wall -o hh hh.o -lm 
```

### 実行

``` powershell
hh > hh.dat
```
