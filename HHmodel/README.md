# HH Model Program

## 環境

使用言語：C / C++14  
エディタ：VSCode

## 環境構築手順

### 0, 準備．

・圧縮・解凍ソフト **7-Zip**のダウンロード．  

### 1, gccコンパイラの導入．

1.1, [Downloads-Mingw-w64](https://www.mingw-w64.org/downloads/)を開く．  
1.2, 圧縮解凍版のgccコンパイラ`x86_64-posix-sjlj`をダウンロードする．  
1.3, 7-Zipを用いて解凍．  
1.4, `mingw64`フォルダを作業環境にコピー．  
1.5, Path設定．  
・スタートメニューを右クリックして**システム**を選択．  
・**システムの詳細設定**を選択．  
・詳細設定タブの**環境変数**を選択．  
・XXXXのユーザ環境変数の**Path**を選択．  
・**新規**ボタンを選択．  
・`C:\~\mingw64\bin`と入力(mingw64フォルダを置いた位置を~に代入．)  
・Enterで確定しOKでウィンドウを閉じる．

### 2, C/C++，Code Rannerをインストール．

2.1, Code Rannerを設定．  
・**Run In Terminal**項目にチェックを入れる．  
・`setting json`を編集．  
->・"clang.executable"を追加  
　・"clang.cxxflags"を追加  
　・"c"と"cpp"を編集  

``` json
    "code-runner.runInTerminal": true,
    "clang.executable": "clang++",
    "clang.cxxflags": [ "-std=c++14"],
    "code-runner.executorMap": {
      "javascript": "node",
      "java": "cd $dir && javac $fileName && java $fileNameWithoutExt",
      
      "c": "cd $dir && gcc $fileName -o $fileNameWithoutExt && $dir$fileNameWithoutExt",
      "zig": "zig run",
      "cpp": "cd $dir && g++ -O3 -std=c++14 $fileName && ./a.out",
      }
```

2.2, C/C++を編集．  
・`Shift + Ctrl + P` でコマンドパレットを開く．  
・**C/C++:Edit Configurations (UI)** を選択．  
・下の方にある C++標準 で **c++14** を選択．  

### 3, gnuplotをインストール．

3.1, [gnuplot homepage](http://www.gnuplot.info/)にアクセス．  
3.2, **Gnuplot XX (current stable)** の最初の項目を選択．(2024/02/08時点:[Release 6.0.0(December 2023)](https://sourceforge.net/projects/gnuplot/files/gnuplot/6.0.0/))  
3.3, インストーラーをダウンロード．  
3.4, インストーラーを起動．  
・日本語でOK．  
・コンポーネントの選択で**日本語対応**にチェック．  
・追加タスクの選択で**実行ファイルのディレクトリをPATHの環境変数に追加する**にチェック．  
3.5, VSCodeから実行できるように設定．  
・Code Rannerの`settings.json`を編集．  
-> ・".plt"を追加

``` json
"code-runner.executorMapByFileExtension": {
    
    ".plt": "gnuplot $fullFileName",
    ".vb": "cd $dir && vbc /nologo $fileName && $dir$fileNameWithoutExt",
    ".vbs": "cscript //Nologo",
    ".scala": "scala",
    },
```

## 実行方法

1, ディレクトリ構成．  
<p align="center">
<img src="https://github.com/c0b2107561/NeuralCircuitSimulation/blob/main/HHmodel/tree.png" width="350px">
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

## 参考
[VSCodeで利用するC++のバージョンを初心者が変更する](https://qiita.com/YuH25/items/be41fce42641a5ce344e)  
[計算科学のためのWindowsセットアップ](https://zenn.dev/ohno/books/356315a0e6437c/viewer/68bd69)  
[WindowsにC言語開発環境Mingw-w64(gccコンパイラ)を導入する](https://dianxnao.com/windows%e3%81%abc%e8%a8%80%e8%aa%9e%e9%96%8b%e7%99%ba%e7%92%b0%e5%a2%83-mingw-w64%ef%bc%88gcc%e3%82%b3%e3%83%b3%e3%83%91%e3%82%a4%e3%83%a9%ef%bc%89%e3%82%92%e5%b0%8e%e5%85%a5%e3%81%99%e3%82%8b/)  
[Visual Studio CodeでC言語プログラミングを始める(Windows編)](https://dianxnao.com/visual-studio-code%E3%81%A7c%E8%A8%80%E8%AA%9E%E3%83%97%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%9F%E3%83%B3%E3%82%B0%E3%82%92%E5%A7%8B%E3%82%81%E3%82%8B%EF%BC%88windows%E7%B7%A8%EF%BC%89/)
