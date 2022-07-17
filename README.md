# Face to Body

## 機能
顔写真を入力すると、体重、身長、BMIを推定する

## アプリケーションへの実装
技育CAMPハッカソンで作成した [Henkenyzer](https://github.com/shimizuyuta/hackathon_vol5) のバックエンドに使用しました

## リポジトリについて
FACE2BMIという論文がもととなっているらしい。顔写真をエンコードして128次元の配列に変換することで、顔の特徴量を作成することができる。実際、face_recognitionのエンコードを使用して、特徴間の距離から同一人物の推定にも使用されている（精度は不明）。これを利用して機械学習により、体重、身長、BMIを推定する。参考にしたリポジトリでは、外国の著名人を学習データとしていたため、日本人の予測ができるようにスクレイピングでデータセットを作成。また、顔のエンコードデータのみでは予測精度が低かったため、顔画像から年齢と性別を推定し、その結果を入力データに統合することで、予測精度を向上。PyCaretによる学習、モデルの保存、予測までを作成したが、herokuにデプロイする際にサイズが大きすぎたため、軽量化するためにsklearnでも実装。

## Solution workflow
- Image ==> Face ==> Face embedding ==> Height<br>(予測精度が低かったのでWeight, BMIから逆算するとよい）

- Image ==> Face ==> Face embedding ==> Weight

- Image ==> Face ==> Face embedding ==> BMI

---

参考：[Face-to-height-weight-BMI-estimation](https://github.com/abhaymise/Face-to-height-weight-BMI-estimation-)
