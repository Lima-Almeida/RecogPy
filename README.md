# RecogPy
Facial/Gesture recognition with python.

At the current state, this application uses webcam live footage to recognize your hand(s).

Output -> [x, x, x, x, x] xx.xx

The vector "[x, x, x, x, x]" represents the states of your 5 fingers, being [thumb, index, middle, ring, pinky]. The state 1 represents a stretched finger and the state 0 represents a bent/flexed finger.

The number "xx.xx" given after the vector represents the angle of your hand.
