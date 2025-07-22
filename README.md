# A Two-stage Approach for Solving Multi-period Cardinality-constraint-based Planning Problem

This repository contains the official implementation of the paper:

> **A Two-stage Approach for Solving Multi-period Cardinality-constraint-based Planning Problem**  
> Our method decomposes the original problem into two stages:  
> (1) candidate selection using reinforcement learning with relation-aware Q-networks,  
> (2) cost-efficient allocation solved via optimization.  
> It is evaluated on scenarios such as sparse index tracking (SIT) and supplier selection and order allocation (SSOA).

## 🚀 Features

- Handles multi-period decision-making with cardinality constraints
- Supports cost-aware reinforcement learning for subset selection
- Integrates optimization (e.g., quadratic programming) for allocation
- Applicable to real-world tasks including SIT and SSOA

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/your_repo_name.git
   cd your_repo_name

## 📦 Running the code

2. Running the code with following command:
   ```bash
   python main.py --dataset dataname --K 5


