---
title: Setup
---

## Setup

## Software

You will need a terminal, Python 3.8+, and the ability to create Python virtual environments.

::::::::::::::::::::::::::::::::::::::: callout

### Installing Python

Python is a popular language for scientific computing, and a frequent choice
for machine learning as well.
To install Python, follow the [Beginner's Guide](https://wiki.python.org/moin/BeginnersGuide/Download) or head straight to the [download page](https://www.python.org/downloads/).

Please set up your python environment at least a day in advance of the workshop.
If you encounter problems with the installation procedure, ask your workshop organizers via e-mail for assistance so
you are ready to go as soon as the workshop begins.

:::::::::::::::::::::::::::::::::::::::::::::::::::

## Packages

You will need the Seaborn, MatPlotLib, Pandas, Numpy and OpenCV packages.

## Directory Setup

Create a new directory for the workshop, then launch a terminal in it:

```bash
mkdir workshop-ml
cd workshop-ml
```

## Creating a new Virtual Environment
We'll install the prerequisites in a virtual environment, to prevent them from cluttering up your Python environment and causing conflicts.
First, create a new directory and ent

::: tab

### venv

Python usually comes with a built-in module called `venv` that allows you to create isolated environments for your
projects.


To create a new virtual environment ("venv") called "intro_ml" for the project, open the terminal (Mac/Linux), Git Bash (Windows) or Anaconda Prompt (Windows), and type one of the below OS-specific options:

```bash
python3 -m venv intro_ml # mac/linux
python -m venv intro_ml # windows
```

(If you're on Linux and this doesn't work, you may need to install venv first. Try running `sudo apt-get install python3-venv` first, then `python3 -m venv intro_ml`)

To activate the environment, run the following OS-specific commands in Terminal (Mac/Linux) or Git Bash (Windows) or Anaconda Prompt (Windows):

* Windows + Git Bash: `source intro_ml/Scripts/activate`
* Windows + Anaconda Prompt: `intro_ml/Scripts/activate`
* Mac/Linux: `source intro_ml/bin/activate`

With the environment activated, install the prerequisites:

```bash
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn
```

### uv

UV is a package manager that allows you to easily create and manage virtual environments. It is a
great alternative to `venv`, `pip`, and `conda` for managing Python environments and packages. It
is a newer tool, written in Rust, and is designed to be fast and efficient.

To install UV, run the following command in your terminal:

```bash
pip install uv
```

To create a new virtual environment in your current directory, run:

```bash
uv venv
```

This will create a new virtual environment in a directory called `.venv` in your current directory.

You should see a message indicating that the virtual environment has been created successfully, and
a command to activate it (this command will vary depending on your operating system):

```bash
Using CPython 3.12.6 interpreter at: C:\python\3.12.6\python.exe
Creating virtual environment at: .venv
Activate with: source .venv/Scripts/activate
```

Run the indicated command to activate the virtual environment. You should now see the name of the
virtual environment in your terminal prompt in parentheses, indicating that it is active:

```bash
user@my_device MINGW64 /c/Users/user/Documents/intro_ml
$ source .venv/Scripts/activate
(intro_ml)
```

Once the environment is activated, you can install the required packages using:

```bash
uv pip install numpy pandas matplotlib seaborn opencv-python scikit-learn
```

:::

## Deactivating/activating environment
To deactivate your virtual environment, simply run `deactivate` in your terminal or prompt. If you close the terminal, Git Bash, or Conda Prompt without deactivating, the environment will automatically close as the session ends. Later, you can reactivate the environment using the "Activate environment" instructions above to continue working. If you want to keep coding in the same terminal but no longer need this environment, it’s best to explicitly deactivate it. This ensures that the software installed for this workshop doesn’t interfere with your default Python setup or other projects.

## Fallback option: cloud environment
If a local installation does not work for you, it is also possible to run this lesson in [Google colab](https://colab.research.google.com/). If you open a jupyter notebook there, the required packages are already pre-installed.
