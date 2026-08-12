import cv2
import os

vid = "/mnt/c/Users/jorda/Downloads/IMG_0298.MOV"
out = "/home/jordan/fencing-mmpose-dev3/local_workspace"
os.makedirs(out, exist_ok=True)
cap = cv2.VideoCapture(vid)
print("opened", cap.isOpened())
print(
    "size",
    int(cap.get(3)),
    int(cap.get(4)),
    "fps",
    cap.get(5),
    "n",
    int(cap.get(7)),
)
for idx in [0, 30, 90, 150, 300]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    if not ok:
        print("fail", idx)
        continue
    p = os.path.join(out, f"frame_{idx}.jpg")
    cv2.imwrite(p, fr)
    print("wrote", p, fr.shape)
cap.release()
