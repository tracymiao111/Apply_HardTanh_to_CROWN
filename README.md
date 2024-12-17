# Apply_HardTanh_to_CROWN

**Abstract:**
This project is for Project 3 of course MICS-5510 in HKUST,GZ. In this work, CROWN and optimizations inspired from alpha CROWN for HardTanh are applied. It is supposed to be noted that this work is extended from a coding problem of UIUC ECE-584 lectured by Prof. Huan Zhang.

**Structure:**

```plaintext
├── README.md
├── __pycache__
│   ├── hardTanh_question.cpython-39.pyc
│   ├── linear.cpython-39.pyc
│   ├── model.cpython-39.pyc
│   └── relu.cpython-39.pyc
├── compare_bounds.py
├── comparison_results.csv
├── comparison_results.png
├── crown.py
├── data1.pth
├── environment.yml
├── hardTanh_question.py
├── linear.py
├── model.py
├── models
│   ├── hardtanh_model.pth
│   └── relu_model.pth
├── plot_bounds.py
├── plot_hardtanh.py
├── relu.py
└── visualize_results.py

**Command to play with it:**

Run CROWN:
```bash
python crown.py -a hardtanh data1.pth --optimize(optional)  # For HardTanh
python crown.py -a relu data1.pth --optimize(optional)      # For ReLU


