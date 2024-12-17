## Apply_HardTanh_to_CROWN

1. **Abstract:**
This project is for Project 3 of course MICS-5510 in HKUST,GZ. In this work, CROWN and optimizations inspired from alpha CROWN for HardTanh are applied. It is important to note that this project extends a coding problem from the UIUC ECE-584 course, taught by Prof. Huan Zhang.
2. **Structure:**
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
```

3. **To play with it:**

run CROWN for HardTanh or Relu:
```plaintext
python crown.py  -a hardtanh(or relu) data1.pth --optimize(optional)
```
compare non-optimized and optimized performance: 
```plaintext
python compare_bounds.py --activation hardtanh --data_file data1.pth --max_eps 0.01 --step_eps 0.001 --output_csv comparison_results.csv
```
plot bounds: 
```plaintext
python plot_bounds.py
```
plot hardtanh: 
```plaintext
python plot_hardtanh.py
```

4. **Prerequisites:**

Check the environment.yml.

