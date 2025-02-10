# MiniHuman: Simulating, Augmenting, and Regulating Virtual Human Behaviors and Responses to Environmental Stimuli

[![PyPI version](https://badge.fury.io/py/minihuman.svg)](https://badge.fury.io/py/minihuman)
[![Downloads](https://pepy.tech/badge/minihuman)](https://pepy.tech/project/minihuman)
[![GitHub Issues](https://img.shields.io/github/issues/songlinxu/MiniHuman.svg)](https://github.com/songlinxu/MiniHuman/issues)



## Introduction
MiniHuman aims to leverage SOTA AI models (LLM agents, RL agents, etc), sensing, and intervention techniques to simulate both **human mind** (mental behaviors) and **physical behaviors**, to support AI and HCI applications in **education**, **health**, **recommendation**, **user experience**, and **human augmentation**.

## News

More models, behaviors, environments, and applications will be released soon!

## Applications

### Education
- LLM-based Student Simulator (e.g., [Classroom Simulacra](https://arxiv.org/abs/2502.02780) [CHI 2025], [EduAgent](https://arxiv.org/abs/2404.07963), [EduTwin](https://arxiv.org/abs/2310.19206))
- Student Behavior Regulation (e.g., [PeerEdu](https://arxiv.org/abs/2312.02358) [CHI 2025])

### Mental Health
- Attention Regulation (e.g., [Eyerofeedback](https://arxiv.org/abs/2307.15172))


### Human Cognitive Augmentation
- Deep Reinforcement Learning Based Cognition Model (e.g., [ReactiveAgent](https://arxiv.org/abs/2301.06216))
- Cognitive Training (e.g., [TimeCare](https://dl.acm.org/doi/10.1145/3544548.3580905) [CHI 2023])

### Climate Change


### Scientific Discovery


### Social Science
- Social Interaction
- Social Network

## Technical Details

### Human Behaviors: Mind + Physical Behaviors
We support a wide range of human behaviors to empower diverse application scenarios. 
- Physiological behaviors
  - Gaze
- Motor behaviors
- Cognitive states
  - Attention
  - Workload
  - ...
- Knowledge states
- Emotion



### Environmental Stimuli
We support a wide range of environmental stimuli to empower diverse application scenarios. 
- Visual stimuli
  - Time Pressure
  - ...
- Auditory stimuli
  - Instructor Voice
  - ...
- Textual stimuli
  - Course Materials
  - ...


### Agent Architecture
- LLM-based Agents
  - Persona Initialization
  - Task Definition
  - Behavior Definition
  - Environment Description
  - Foundation Model
- RL-based Agents
  - Action Space
  - Observation Space
  - Optimization Policy

## Citation

If you use this repository, please cite it as follows:

Songlin Xu, Xinyu Zhang. (2024). MiniHuman-Toolkit (Version 0.1.2). GitHub. https://github.com/songlinxu/MiniHuman-Toolkit

Alternatively, use the following BibTeX entry:

```bibtex
@misc{songlin2024minihuman,
  author = {Songlin Xu, Xinyu Zhang},
  title = {minihuman},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/songlinxu/MiniHuman-Toolkit}},
  version = {0.1.2}
}


