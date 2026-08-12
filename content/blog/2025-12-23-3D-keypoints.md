---
title: Lifting 2D to 3D keypoints 
slug: lifting-2d-to-3d-keypoints 
date: 2025-12-23
published: true
tags: [announcement, project]
summary: Improvment with 3D keypoints 
---

## 3d Lifting

For poses I have been using an open-source library called **MMPose**. This library has given me access to more accurate 2d models like **ViTPose** and more importantly, a **3d lifting model called Motionbert**. After the 2d data extraction phase done by ViTPose, Motionbert lifts the captured 2d sequences of keypoints into sequences of 3d keypoints.

Nevertheless, to keep my application accessible to a variety of people, I have **not compromised the program's monocularity**, making sure that only one phone is necessary to film. Since there is no way to gauge depth monocularly, the **3d sequences have been limited to their own space**. While Motionbert provides the poses of the fencers, it is nearly impossible to get the fencers into a shared 3d space accurately. Still, the **angle-invariant nature of 3d keypoints** should prove helpful in training an **unbiased model**.
