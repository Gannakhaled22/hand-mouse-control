import mediapipe as mp

print(" MediaPipe version:", mp.__version__)
print(" Available attributes:", [x for x in dir(mp) if not x.startswith('_')])


try:
    mp_hands = mp.solutions.hands
    print(" mp.solutions.hands works!")
except Exception as e:
    print(" mp.solutions.hands failed:", e)

try:
    mp_drawing = mp.solutions.drawing_utils
    print(" mp.solutions.drawing_utils works!")
except Exception as e:
    print(" mp.solutions.drawing_utils failed:", e)

print(" MediaPipe test complete!")
