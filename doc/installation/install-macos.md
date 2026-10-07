## How to install Iris on macOS

This guide installs Iris on a Mac. On a Mac, Iris runs on its local engine, without Spark or Java. With Spark support installed, it also runs on its Spark engine, with Spark running on the Mac only, without a cluster. All eight parts of the [tutorial](../tutorial.md) run on it.

The installation produces a Python environment, the `.venv` folder, inside the `iris` folder. Nothing is installed system-wide by the installer, and nothing outside the `iris` folder is modified. The prerequisites in Step 1, Python and, for Spark support, Java, are installed separately, before the installer runs.

This guide was tested on a Mac with Apple silicon running macOS 26.4, with Python 3.12.14 and Java 17.0.20. The build of `py_stringmatching` on an Intel Mac has not been tested.

---

## Step 1: Check the prerequisites

Iris needs Python 3.10, 3.11 or 3.12. On a Mac with Apple silicon, such as an M1 or later, every dependency installs from a prebuilt package. On a Mac with an Intel processor, one dependency, `py_stringmatching`, is built from source, which needs the Xcode Command Line Tools. Spark support, an optional part of Iris that adds PySpark, also needs Java 17.

### The Xcode Command Line Tools, on an Intel Mac only

Skip this step on a Mac with Apple silicon. On an Intel Mac, open Terminal (in Applications > Utilities) and execute:

```
xcode-select -p
```

You should see a path such as:

```
/Library/Developer/CommandLineTools
```

If you see `xcode-select: error: unable to get active developer directory`, execute `xcode-select --install`, click **Install** in the window that appears, and wait for the installation to finish, which often takes a few minutes.

### Python

The `python3` that comes with the Command Line Tools is Python 3.9, which Iris does not support. Execute:

```
python3.12 --version
```

You should see:

```
Python 3.12.14
```

or another 3.12 version. If you see `command not found: python3.12`, install Python 3.12 in one of two ways:

- From [python.org](https://www.python.org/downloads/macos/): download the macOS installer of the latest Python 3.12 release and run it.
- With Homebrew: `brew install python@3.12`. Homebrew's `python3` is Python 3.13 or later, which Iris does not support, so install `python@3.12` by name.

The installer uses the first of `python3.12`, `python3.11`, `python3.10` and `python3` that is a supported version.

### Java, for Spark support only

Skip this step if you will not run Iris on Spark. Otherwise, execute:

```
java -version
```

You should see version 17:

```
openjdk version "17.0.20" 2026-07-21
```

If Java is missing or is another version, download the Java 17 installer for macOS from [Adoptium](https://adoptium.net/temurin/releases/?version=17) and run it, or execute `brew install --cask temurin@17`. Spark 3.5 runs on Java 8, 11 and 17 only. A newer Java, such as 21, makes Spark fail later with errors that look unrelated, so the installer refuses it.

---

## Step 2: Download Iris

Open [the repository page](https://github.com/relixa-ai/iris) in your browser, click the green **Code** button, and click **Download ZIP**. Unzip the file in your **Downloads** folder and rename the folder it creates to `iris`. If you have `git`, you can instead execute `git clone https://github.com/relixa-ai/iris.git`. Either way, the folder contains:

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

In Terminal, go to the `iris` folder. Type `cd `, drag the folder from Finder into the Terminal window, and press Return. Then execute:

```
bash install.sh
```

To install Iris with Spark support, execute `bash install.sh --spark` instead.

This command does the following.

- **Checks Python**, the Command Line Tools on an Intel Mac, and, with `--spark`, Java.
- **Creates the folder `.venv`** inside the `iris` folder, which is a Python environment separate from the system's Python.
- **Installs Iris and its dependencies** into `.venv` with `pip`, from the Python Package Index ([pypi.org](https://pypi.org)), at the versions listed in `constraints.txt`. The dependencies are NumPy, pandas, PyArrow, SciPy, scikit-learn, XGBoost, Numba, mmh3, py_stringmatching, joblib, threadpoolctl and tqdm, and with `--spark` also PySpark, which contains Spark. Each is a prebuilt package, except `py_stringmatching` on an Intel Mac, which pip builds from source. This step takes a few minutes.
- **Runs a short test**, `iris smoke`, which creates the features of two small built-in tables, featurizes their 64 pairs, and trains and applies a matcher. With `--spark` it runs the same test on Spark.

It does not use `sudo`, does not ask for your password, and does not modify anything outside the `iris` folder. You should see output like the following, with your own paths. It took 55 seconds on a Mac with Apple silicon.

```
[1/6] checking Python
      OK: Python 3.12 at /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12
[2/6] checking the build tools
      skipped: every dependency installs from a prebuilt package here
[3/6] checking Java
      skipped: the local engine needs no Java (bash install.sh --spark adds Spark)
[4/6] creating .venv
      OK: .venv/ created with Python 3.12
[5/6] installing iris and its dependencies with pip, from PyPI (a few minutes)
      OK: iris 0.1.0 installed
[6/6] smoke test
iris 0.1.0
python    3.12.14
engine    local, up to 18 worker processes
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

With `--spark`, step 3 prints `OK: Java 17.0.20`, and the smoke test runs a second time on Spark, which prints `java` and `spark` lines and Spark's own startup messages.

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

---

## Troubleshooting

| What you see | What it means | What to do |
|---|---|---|
| `ERROR: Python 3.9 is not supported.` | Only the Python of the Command Line Tools was found. | Install Python 3.12 as in Step 1. |
| `ERROR: Python 3.13 is not supported.` | Only Homebrew's default Python was found. | `brew install python@3.12`, or point the installer at a supported Python: `PYTHON=/path/to/python3.12 bash install.sh`. |
| `ERROR: the Xcode Command Line Tools are not installed.` | On an Intel Mac, no compiler is available to build `py_stringmatching`. | `xcode-select --install`, then run the installer again. |
| `ERROR: no Java runtime found.` | `--spark` was given and Java is not installed. | Install Java 17 as in Step 1. |
| `ERROR: Java 21.0.5 is not supported by Spark 3.5.` | The default Java is too new for Spark. | Install Java 17 and set `JAVA_HOME` to it: `export JAVA_HOME=$(/usr/libexec/java_home -v 17)`. |
| `no such file or directory: install.sh` | Terminal is not in the `iris` folder. | Repeat the start of Step 3. |
| `command not found: iris` | The environment is not active in this Terminal window. | Run `source .venv/bin/activate` from the `iris` folder. Your prompt then shows `(.venv)`. |
| Errors mentioning the network, or `Could not find a version` | A package could not be downloaded. | Check the internet connection and run the installer again. Running it twice is safe. |

---

## How to uninstall

Everything the installer created is in the `iris` folder. To remove the Python environment, execute `rm -rf .venv` in that folder. To remove Iris completely, drag the `iris` folder to the Trash. Python, Java and the Command Line Tools from Step 1 can be kept or removed separately.
