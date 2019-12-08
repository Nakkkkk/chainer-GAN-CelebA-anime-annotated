# GAN CelebA from Anima img

# 改造前
``` sh

https://github.com/xkumiyu/chainer-GAN-CelebA

```

# 概要
``` sh

リアルの人間画像→アニメの人間画像

は、あったので、

アニメの人間画像→リアルの人間画像

を、作った。

普通のGANの入力をノイズからアニメ画像に変更した。モード崩壊を防ぐためにMinibatch_Discriminationが有効だった。

学習時間はだいたい５時間（500epoch, gtx1080_8GB）。

趣味用なので、パスとかめちゃくちゃです。

```

# 使い方
``` sh
＜準備＞

人間顔データセット：http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
アニメ顔データセット：http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/

から、ダウンロード。

＜学習＞

$ python train_gen.py

＜生成＞

$ python inference.py

```

# 反省点
``` sh

・GANは評価がしにくい。完全に見た目評価。
・それっぽいものはたまに（５回に一回）出てくる。
・アノテーションの効果は微妙。
・アニメ画像に影とかが加わって暗くなると、生成画像は黒・茶髪の人が生成されやすい。
・CycleGANとか使いたいなー。

```
