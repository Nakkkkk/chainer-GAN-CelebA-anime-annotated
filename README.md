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

# 結果
``` sh

＜成功例？＞
アニメ画像
![in_image](https://user-images.githubusercontent.com/38216393/70388681-53fe9b00-19f8-11ea-8f4e-affde54eec8f.png)
人間画像
![out_image](https://user-images.githubusercontent.com/38216393/70388697-9e801780-19f8-11ea-954a-e0becb6617e4.png)

顔の向き、金髪（ほんのりピンク）、性別といった特徴において、アニメと人間がよく一致していた。

＜失敗例？＞
アニメ画像
![in_image](https://user-images.githubusercontent.com/38216393/70388706-b8215f00-19f8-11ea-9c89-4dfa845da0d3.png)
人間画像
![out_image](https://user-images.githubusercontent.com/38216393/70388710-c7a0a800-19f8-11ea-9d8f-6742ec8d465b.png)

前髪の形が似ている。しかし、アニメキャラに影が入った影響で人間は黒髪の模様。

```


# 反省点
``` sh

・GANは評価がしにくい。完全に見た目評価。
・それっぽいものはたまに（５回に一回）出てくる。
・アノテーションの効果は微妙。
・アニメ画像に影とかが加わって暗くなると、生成画像は黒・茶髪の人が生成されやすい。
・CycleGANとか使いたいなー。

```
