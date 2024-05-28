# starline
**St**rict coloring m**a**chine fo**r** **line** drawings.


![image](https://github.com/mattyamonaca/starline/assets/48423148/eae07a6e-9c7b-4292-8c70-dac8ec8eeb7b)


https://github.com/mattyamonaca/starline/assets/48423148/8199c65c-a19f-42e9-aab7-df5ed6ef5b4c

# Installation
```
git clone https://github.com/mattyamonaca/starline.git
cd starline
conda create -n starline python=3.10
conda activate starline
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

# Usage
- ```python app.py```
- Input the line drawing you wish to color (The background should be transparent).
- Input a prompt describing the color you want to add.

- 背景を透過した状態で線画を入力します
- 付けたい色を説明するプロンプトを入力します

# Precautions
- Image size 1024 x 1024 is recommended.
- Aliasing is a beta version.
- Areas finely surrounded by line drawings cannot be colored.

- 画像サイズは1024×1024を推奨します
- エイリアス処理はβ版です。より線画に忠実であることを求める場合は2値線画を推奨します
- 線画で細かく囲まれた部分は着色できません。着色できない部分は透過した状態で出力されます。
