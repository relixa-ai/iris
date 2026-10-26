## How to install Iris on Linux

This guide installs Iris on one Linux machine. On it, Iris runs on its local engine. With Spark support installed, it also runs on its Spark engine, with Spark running on this machine only. All eight parts of the [tutorial](../tutorial.md) run on it.

The installation produces a Python environment, the `.venv` directory, inside the `iris` directory. The prerequisites in Step 1 are installed with the system's package manager.

This guide was tested on Ubuntu 24.04, on x86 and on ARM, with Python 3.12.3 and OpenJDK 17.0.20.

---

## Step 1: Check the prerequisites

Iris needs Python 3.10, 3.11 or 3.12, with its `venv` module. Every dependency installs from a prebuilt package. Spark support, an optional part of Iris that adds PySpark, also needs Java 17.

### Python

Execute:

```
python3 --version
```

You should see 3.10, 3.11 or 3.12, for example:

```
Python 3.12.3
```

The installer uses the first of `python3.12`, `python3.11`, `python3.10` and `python3` that is a supported version, so a machine whose `python3` is 3.13 can still install Iris when one of the others is present.

| What you see | What it means | What to do |
|---|---|---|
| `Python 3.13.1`, or a version below 3.10 | No supported Python is the default. | Install Python 3.12 from your distribution's packages, for example `sudo apt-get install python3.12 python3.12-venv python3.12-dev`. |
| `python3: command not found` | Python is not installed. | Install it from your distribution's packages. |

### The venv module

Debian and Ubuntu package Python's `venv` module separately. Execute:

```
sudo apt-get install python3-venv
```

On Fedora, Red Hat and Rocky Linux the module is part of Python, so skip this step.

### Java, for Spark support only

Skip this step if you will not run Iris on Spark. Otherwise, on Debian or Ubuntu, execute:

```
sudo apt-get install openjdk-17-jdk-headless
java -version
```

You should see version 17:

```
openjdk version "17.0.20.1" 2026-08-18
```

Spark 3.5 runs on Java 8, 11 and 17 only. A newer Java, such as 21, makes Spark fail later with errors that look unrelated, so the installer refuses it.

---

## Step 2: Download Iris

Clone the repository with `git`:

```
git clone https://github.com/relixa-ai/iris.git
cd iris
```

You can instead download the ZIP file from [the repository page](https://github.com/relixa-ai/iris), with the green **Code** button and **Download ZIP**, unzip it, and rename the directory to `iris`. Either way, the directory contains:

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

---

## Step 3: Run the installer

From the `iris` directory, execute:

```
bash install.sh
```

To install Iris with Spark support, execute `bash install.sh --spark` instead.

This command does the following.

- **Checks Python** and, with `--spark`, Java.
- **Creates the directory `.venv`** inside the `iris` directory, which is a Python environment separate from the system's Python.
- **Installs Iris and its dependencies** into `.venv` with `pip`, from the Python Package Index ([pypi.org](https://pypi.org)), at the versions listed in `constraints.txt`. The dependencies are NumPy, pandas, PyArrow, SciPy, scikit-learn, XGBoost, Numba, mmh3, `py_stringmatching`, joblib, threadpoolctl and tqdm, and with `--spark` also PySpark, which contains Spark. Each is a prebuilt package. This step takes a few minutes.
- **Runs a short test**, `iris smoke`, which creates the features of two small built-in tables, featurizes their 64 pairs, and trains and applies a matcher. With `--spark` it runs the same test on Spark.

You should see output like the following:

```
[1/6] checking Python
      OK: Python 3.12 at /usr/bin/python3.12
[2/6] checking the build tools
      skipped: every dependency installs from a prebuilt package here
[3/6] checking Java
      skipped: Java is checked by bash install.sh --spark, which adds Spark support
[4/6] creating .venv
      OK: .venv/ created with Python 3.12
[5/6] installing iris and its dependencies with pip, from PyPI (a few minutes)
      OK: iris 0.3.0 installed
[6/6] smoke test
iris 0.3.0
python    3.12.3
engine    local, up to 4 worker processes
features  13
featurize 64 vectors x 13
matcher   8/8 known matches recovered

OK: iris works on this machine.

  ┌────────────────────────────────────────────────────────┐
  │  iris is installed.                                    │
  │                                                        │
  │  Turn the environment on:                              │
  │      source .venv/bin/activate                         │
  │                                                        │
  │  Run the first example:                                │
  │      python examples/run_passive.py                    │
  │                                                        │
  │  The tutorial is doc/tutorial.md.                      │
  └────────────────────────────────────────────────────────┘
```

With `--spark`, step `[3/6]` prints the Java version, and the smoke test runs a second time on Spark, which prints `java` and `spark` lines and Spark's own startup messages.

If you see `OK: iris works on this machine.` and the closing box, Iris is installed. Activate the environment and run the first part of the tutorial:

```
source .venv/bin/activate
python examples/run_passive.py
```

---

## Checking the installation

`iris smoke` runs the smoke test again, and `iris info` shows the versions, the memory and the cores Iris sees, the number of worker processes it will use, and whether Spark and Java are present:

```
source .venv/bin/activate
iris info
```

```
iris      0.3.0
python    3.12.3
machine   31.3 GiB, 4 cores (cgroup (31.3 GiB / 4 cores; machine reports 31.3 GiB / 18 cores))
local     up to 4 worker processes, scratch in /tmp
spark     not installed (pip install 'iris[spark]')
```

Inside a container, a Slurm job or a systemd service, Iris counts the memory and the cores that it is allowed, not those of the machine, as `cgroup` in the `machine` line shows. When `IRIS_MEMORY_GB` is set lower, the parentheses begin with `setting IRIS_MEMORY_GB` instead.

---

## Troubleshooting

| What you see | What it means | What to do |
|---|---|---|
| `ERROR: Python 3.13 is not supported.` | No Python 3.10 to 3.12 was found. | Install Python 3.12 as in Step 1, or point the installer at one: `PYTHON=/path/to/python3.12 bash install.sh`. |
| `ERROR: Python 3.12 cannot create a virtual environment: its venv module is not installed.` | Debian and Ubuntu package the `venv` module separately. | `sudo apt-get install python3-venv`, or `python3.12-venv`. |
| `ERROR: no Java runtime found.` | `--spark` was given and Java is not installed. | Install Java 17 as in Step 1, or set `JAVA_HOME` to a Java 17 installation. |
| `ERROR: Java 21.0.5 is not supported by Spark 3.5.` | The default Java is too new for Spark. | Install Java 17 and set `JAVA_HOME` to it, for example `export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64`. |
| `iris: command not found` | The environment is not active in this terminal. | Run `source .venv/bin/activate` from the `iris` directory. |
| Errors mentioning the network, or `Could not find a version` | A package could not be downloaded. | Check the internet connection and run the installer again. |

---

## How to uninstall

Everything the installer created is in the `iris` directory. To remove the Python environment, execute `rm -rf .venv` in that directory. To remove Iris completely, delete the `iris` directory. The packages installed in Step 1 can be kept, or removed with the system's package manager.
