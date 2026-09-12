# Multi-Stage Keyword Spotting

This repository implements a low-latency real-time audio processing pipeline for keyword spotting (KWS). Incoming audio flows through a multi-stage cascading architecture designed to minimize computational overhead during quiet periods.

The pipeline begins with an RMS Energy Filter that drops silent frames before invoking neural models. When audio energy passes the initial threshold, a lightweight binary [Keyword Transformer](https://arxiv.org/abs/2104.00769) model acts as a low-power wake-word detector. Once valid speech is detected, a higher-capacity multi-class [Keyword Transformer](https://arxiv.org/abs/2104.00769) performs keyword classification.

The system was trained and evaluated using the [Google Speech Commands](https://arxiv.org/abs/1804.03209) dataset.
