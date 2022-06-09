# 深度學習課程期末專案-AI Explainability 360 (v0.2.1) 
#### Supported Configurations:

| OS      | Python version |
| ------- | -------------- |
| macOS   | 3.6  |
| Ubuntu  | 3.6  |
| Windows | 3.6  |

# Installation

create a new Python 3.6 environment, run:
```bash
conda create --name aix360 python=3.6
conda activate aix360
```

Clone the latest version of this repository:

```bash
(aix360)$ git clone https://github.com/rayeh1063007/DL_final_AIX360.git
```

```bash
(aix360)$ pip install cvxpy-1.0.31.tar.gz .
```

Then, navigate to the root directory of the project which contains `setup.py` file and run:

```bash
(aix360)$ pip install -e .
```

# 執行方式

1. 下載貓狗data: https://drive.google.com/drive/folders/11V_oJm8_fF2MAV8eBTWSKWQ1A6-BUvz0?usp=sharing

2. 修改 aix_catdog.py 或 Group7_aix.ipynb 內的資料集路徑(line:15、16)

3. 修改 aix_catdog.py 或 Group7_aix.ipynb 內的解釋圖片ID(line:18)

4. 執行 aix_catdog.py 或 Group7_aix.ipynb
