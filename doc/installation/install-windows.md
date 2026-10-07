## How to install Iris on Windows

This guide installs Iris on a Windows machine. With Option B below, Iris runs on its local engine only, without Spark or Java, and Parts 1 to 6 of the [tutorial](../tutorial.md) run. With Option A, Iris runs in Ubuntu under Windows as on a Linux machine, and with Spark support all eight parts run.

There are two ways to install Iris on Windows. The first runs Iris in Ubuntu under the Windows Subsystem for Linux (WSL), and follows the [Linux guide](install-linux.md), which has been tested on Ubuntu 24.04. The second installs Iris in Windows itself, with Python for Windows. **This guide has not yet been tested on Windows.** If a step fails, please open an issue with the full output of the step.

The installation produces a Python environment, the `.venv` folder, inside the `iris` folder. Nothing is installed system-wide, and nothing outside the `iris` folder is modified, apart from the prerequisites in Step 1.

---

## Option A: Ubuntu under WSL

In PowerShell, run as administrator, execute:

```
wsl --install -d Ubuntu-24.04
```

Restart Windows when asked, open **Ubuntu** from the Start menu, and create a user name and a password when asked. Then follow the [Linux guide](install-linux.md) in the Ubuntu window, from its Step 1. The `iris` folder is then in Ubuntu's file system, and Iris can be installed with Spark support as on Linux.

---

## Option B: Windows itself

### Step 1: Check the prerequisites

Iris needs Python 3.10, 3.11 or 3.12. Every dependency installs from a prebuilt package, so no compiler is needed. This option does not cover Spark support, which on Windows also needs Hadoop's Windows program `winutils.exe` and settings that this guide does not install. Use Option A for Spark.

#### Python

Open PowerShell and execute:

```
py -3.12 --version
```

You should see a 3.12 version, such as `Python 3.12.10`. If you see an error, download the Windows installer of the latest Python 3.12 release from [python.org](https://www.python.org/downloads/windows/), run it, and select **Add python.exe to PATH** on its first page.

### Step 2: Download Iris

Open [the repository page](https://github.com/relixa-ai/iris) in your browser, click the green **Code** button, and click **Download ZIP**. Extract the ZIP file, and rename the folder it creates to `iris`. The folder contains:

```
README.md
constraints.txt
data
deploy
doc
examples
install.sh
pyproject.toml
python
```

### Step 3: Install Iris

`install.sh` is a script for Linux and macOS. On Windows, the same steps are run one at a time. In PowerShell, go to the `iris` folder, for example with `cd $HOME\Downloads\iris`, and execute:

```
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e . -c constraints.txt
iris smoke
```

These commands do the following.

- **Create the folder `.venv`** inside the `iris` folder, which is a Python environment separate from the system's Python, and activate it in this PowerShell window.
- **Install Iris and its dependencies** into `.venv` with `pip`, from the Python Package Index ([pypi.org](https://pypi.org)), at the versions listed in `constraints.txt`. The dependencies are NumPy, pandas, PyArrow, SciPy, scikit-learn, XGBoost, Numba, mmh3, py_stringmatching, joblib, threadpoolctl and tqdm, each a prebuilt package. This takes a few minutes.
- **Run a short test**, `iris smoke`, which creates the features of two small built-in tables, featurizes their 64 pairs, and trains and applies a matcher. It ends with `OK: iris works on this machine.`

Then run the first part of the tutorial:

```
python examples\run_passive.py
```

---

## Troubleshooting

| What you see | What it means | What to do |
|---|---|---|
| `.venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system` | PowerShell does not allow scripts by default. | Execute `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again. |
| `No suitable Python runtime found` | Python 3.12 is not installed. | Install it as in Step 1 of Option B. |
| `'iris' is not recognized` | The environment is not active in this PowerShell window. | Execute `.venv\Scripts\Activate.ps1` from the `iris` folder. |

---

## How to uninstall

Everything the installation created is in the `iris` folder. To remove the Python environment, delete the `.venv` folder. To remove Iris completely, delete the `iris` folder. Python from Step 1 can be removed from **Settings > Apps**. Under WSL, the `iris` folder is in Ubuntu, and the Ubuntu installation can be removed with `wsl --unregister Ubuntu-24.04`.
