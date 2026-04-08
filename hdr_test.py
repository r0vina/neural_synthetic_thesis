import cv2
import numpy as np

hdr = cv2.imread("hdr.hdr", cv2.IMREAD_UNCHANGED).astype(np.float32)

tonemap = cv2.createTonemapReinhard(gamma=2.2)
ldr = tonemap.process(hdr.copy())
ldr8 = np.clip(ldr * 255, 0, 255).astype(np.uint8)
cv2.imwrite("preview.png", ldr8)


tm = cv2.createTonemapDrago(gamma=2.2, saturation=1.0, bias=0.85)
ldr = tm.process(hdr.copy())
ldr8 = np.clip(ldr * 255, 0, 255).astype(np.uint8)
cv2.imwrite("preview_drago.png", ldr8)