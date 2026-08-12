---
title: Bellguard Detection
slug: Bellguard Detection
date: 2026-4-10
published: true
tags: [testing]
summary: Training Bellguard Keypoint
---

## Why

Previously, the model for detecting where touches landed was trained on normal COCO keypoints. These keypoints are the standard for pose detection and includes 17 different keypoints for the body. Still, fencing is a sport that requires a lot of wrist movement. Bending of the wrist could make a touch go from a chest touch to a leg touch. Normal COCO keypoints only have wrist keypoints, not hand. Additionally, using an extra hand model could be problematic, since the bellguard often covers things like fingers. That is why I have decided to train an extra keypoint for the bellguard. The theory is that instead of using the angle that the elbow and wrist make, I could use the angle that the wrist and bellguard make, making the model more accurate. Additionally, the bellguard changes distances away from the wrists of the fencers depending on grip style, something that would be alleviated with a bellguard keypoint.

## Initial Solution

In order to train bellguard detection, I first had to annotate images with bellguard. Luckily I had already a few saved bouts on my computer. I used a program to extract individual random frames from each video, then I used an app called CVAT to quickly annotate the bellguards in the frame. After this, I ran the normal model on the saved frames to get the 17 other keypoints. Merged, I had a total of 18 keypoints per fencer. Now, I just had to train an 18 keypoints model, which I could follow the MMPose tutorial and use the ViTPose backbone for.

## Problems

1. At first I followed the standard MMPose architecture for training keypoints. While this seemed to work at first, I realized that the other 17 keypoints other than the bellguard were losing accuracy. This was because I had a limited data set, and I was essentially retraining all 17 keypoints, along with the bellguard.

2. The next issue is implementing the bellguard keypoint effectively into a detection model.

## Solutions & Current Progress

1. Initially, I thought the first problem meant that I had to use a completely seperate model for the bellguard, because the keypoints would always learn together. This is because the training was trying to mimimize total error rather than focusing on the bellguard keypoint. Later, I realized that I could mostly freeze the 17 other keypoints that come with the ViTPose model, causing them to minimally shift and force the model to focus on learning the bellguard.

2. The second issue is ongoing. Since the bellguard is not super stable, I have been only using them in models if their confidence score is greater than 0.85. I am currently trying two different model types. One is a having a small vision model around the bellguard to look at the angle or where the bellguard is pointing. Still, anything vision related is prone to overfitting. The second is just using 2d keypoints. This also has a risk of overfitting, but it is not as bad if I have a wide enough variety of data.