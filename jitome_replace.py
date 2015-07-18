from PIL import Image
import numpy as np
import cv2

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    # 読み込み画像サイズを制限
    size = (640,480)
    ret = cap.set(3,size[0])
    ret = cap.set(4,size[1])

    # じとめ画像を取得
    jitome = Image.open("jitome.png")
    jitome_str = Image.open("jitome2.png")

    while True:
        # カメラ読み込み
        ret, im = cap.read()
        # 目探索用の機械学習ファイルを取得
        cascade = cv2.CascadeClassifier("/usr/local/Cellar/opencv3/3.0.0/share/OpenCV/haarcascades/haarcascade_eye.xml")
        # 目探索(画像,縮小スケール,最低矩形数)
        eye = cascade.detectMultiScale(im, 1.1, 3)

        # 目が２つより多かったらリトライ
        #if len(eye) > 2:
        #    continue

        # PILに変換
        im_pil = Image.fromarray(im)
        im_pil = im_pil.convert('RGBA')

        # 透過キャンパスを生成
        c = Image.new('RGBA', im_pil.size, (255, 255,255, 0))

        c.paste(jitome_str, (400,0))

        # 目検出した部分を長方形で囲う
        for (x, y, w, h) in eye:
            # cv2.rectangle(im, (x, y),(x+w, y+h),(0, 50, 255), 3)
            w = int(w * 1.5)
            h = int(h * 1.5)
            # じとめをリサイズ
            jitome_resize = jitome.resize((w,h))
            # 透過キャンパスにじとめを貼り付け
            c.paste(jitome_resize, (x,y))

        # 合成
        im_pil = Image.alpha_composite(im_pil, c)

        # OpenCVオブジェクトに変換
        im = np.asarray(im_pil)
            
        # 画像表示
        cv2.imshow("Show Image",im)
        
        # キー入力待機
        if cv2.waitKey(1) >= 0:
            break
    # 画像保存
    cv2.imwrite("test2.jpg",im)
    cv2.destroyAllWindows()
