import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize camera and set properties
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize hand detector
detector = HandDetector(detectionCon=0.7)

# Load the image once
img1 = cv2.imread('one.jpg')
startDis = None
scale = 0
cx, cy = 200, 200

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if len(hands) == 2:
        hand1 = hands[0]
        hand2 = hands[1]

        hand1_fingers = detector.fingersUp(hand1)
        hand2_fingers = detector.fingersUp(hand2)

        # Check for fists (all fingers down)
        if hand1_fingers == [0, 0, 0, 0, 0] and hand2_fingers == [0, 0, 0, 0, 0]:
            lmList1 = hand1["lmList"]
            lmList2 = hand2["lmList"]

            if startDis is None:
                length, info, img = detector.findDistance(hand1["center"], hand2["center"], img)
                startDis = length

            length, info, img = detector.findDistance(hand1["center"], hand2["center"], img)
            scale = int((length - startDis) // 2)
            cx, cy = info[4:]

    else:
        startDis = None

    try:
        h1, w1, _ = img1.shape
        newH, newW = ((h1 + scale) // 2) * 2, ((w1 + scale) // 2) * 2
        resized_img1 = cv2.resize(img1, (newW, newH))
        img[cy - newH // 2:cy + newH // 2, cx - newW // 2:cx + newW // 2] = resized_img1
    except Exception as e:
        print(e)
        pass

    cv2.imshow("Problem Solve", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
