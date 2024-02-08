# NeuralCircuitSimulation Exection Preparation

## 環境
使用言語：C / C++14  
エディタ：VSCode

## 手順
1. 拡張機能で”C/C++”，”Code Ranner”をインストール．
2. Code Rannerを設定．  
   2.1. `setting json`を編集．

```
"clang.executable": "clang++",
"clang.cxxflags": [ "-std=c++14"],
"code-runner.executorMap": {
  
  "cpp": "cd $dir && g++ -O3 -std=c++14 $fileName && ./a.out",
  ... }
```

3. C/C++を編集．  
   3.1. `Shift + Ctrl + P` でコマンドパレットを開く．  
   3.2. ・**C/C++:Edit Configurations (UI)** を選択．  
   3.3. 下の方にある C++標準 で **c++14** を選択．  
