# UNO-Computer-Vision-Game

A computer vision project. Using python and cv2 as computer vision library.

The goal is to play the popular UNO cards game against a computer, which detects the cards in front of it's webcam.

We've made our own training data library of UNO cards.

At the moment is in working condition:
* color detection on single uno cards
* detection and filtering, where cards on the play area lay.
* cutting the detected cards for further processing
* feature detection with a machine learning approach
* returning detected card (quite accurate)
* saving the trained model for faster startup speed

Missing:
* UNO game logic
* color detection on filtered cards from webcam image
