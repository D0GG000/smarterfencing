#!/usr/bin/env python3
import numpy as np
from scoreboard_tracker import ScoreboardTracker

img = np.zeros((480, 640, 3), dtype=np.uint8)
tracker = ScoreboardTracker(img, (100, 100, 180, 140))
box = tracker.update(img)
snap = tracker.snapshot()
print("backend", tracker.backend.value)
print("box", box)
print("provider", snap.get("provider"))
