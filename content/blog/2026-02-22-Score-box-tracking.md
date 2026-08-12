---
title: Score Box Tracking
slug: score-box-tracking
date: 2026-2-22
published: true
tags: [Future]
summary: Future features to be added
---

## Problem

In Fencing, it is sometimes hard to film a bout without moving the camera. Currently, the camera has to be completely still for the scoring detection to work. This is because the detection system checks for changes in color of the score boxes in the video.

## Potential Solutions

Right now, I can think of two potential solutions. The first of which is adding motion detection to counteract the camera panning. This way, the selected points for the light would move with the camera. Nevertheless, this solution would still be susceptible to occlusion, where the light becomes not visible at all.

The second solution would be to add touch detection without the light. This solution would be much more difficult to implement, but it would be more robust. With this solution, visibility of the light would not matter, allowing for a greater variety of filming angles to be used. The challenge with this solution would be training a model to become accurate enough to estimate the exact frame of the touch, especially with the motion blur of a phone camera.