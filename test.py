import cv2
import time

# Index de la cam�ra int�gr�e
CAM_INDEX = 0  # 0 = premi�re cam�ra, change si tu as plusieurs cam�ras

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print(f"Impossible d'ouvrir la cam�ra {CAM_INDEX}")
    exit(1)

print("Cam�ra ouverte avec succ�s ! Appuie sur 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Impossible de lire l'image de la cam�ra")
        break

    # Affiche le flux vid�o
    cv2.imshow("ADAS Test Cam�ra", frame)

    # Quitte avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
