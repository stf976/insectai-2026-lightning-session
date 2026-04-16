# PyTorch Lightning Training Session, InsectAI Meetup 2026

This is the code repository for the PyTorch Lightning Training Session at the InsectAI Meetup in Niš, Serbia, 22–23 April 2026.

Participants are encouraged to setup a Python development environment and clone this repository on their laptops in advance. Follow installation instructions to below for setup.

However, this is not required, and everyone is welcome to join the training session, with or without a laptop or Python setup.

Please keep in mind that there will be no time for software setup or troubleshooting assistance during the training session. As the main sources of installation support please use the respective guides, instructions and forums provided in the links below.

If you find any bugs and omissions, or have questions about the provided code and instructions, please let me know on GitHub, Slack, or in person!

Currently, this repository only contains installation instructions and the PyTorch example that we will be converting to PyTorch Lightning. The full code will be uploaded before the training session.

See you at InsectAI Meetup in Niš, Serbia, 22–23 April 2026!

## Install Python & pip

To run the provided code, you need a modern version of Python (3.10 or later should be fine).

In addition, the Python package manager `pip` is required.

PyTorch has a short guide on how to install Python and `pip` on Linux, Mac, and Windows: https://pytorch.org/get-started/locally/

Do not install PyTorch just yet! We will use a virtual environment for that.

## Clone Code Repository

Install `git` to clone this repository. Instructions for Linux, Mac, and Windows are at: https://git-scm.com/install/

On a command line, in a directory to contain the source code, run:

`git clone https://github.com/stf976/insectai-2026-lightning-session.git`

## Install Virtual Environment

It is recommended to use a Python virtual environment for each project. This allows for:
- easier dependency management,
- clean uninstalls,
- avoids overwriting system packages,
- and makes dependencies easier to share (via `requirements.txt`)

The Python `venv` package is the recommended way to create a virtual environment. On Windows, it should come with your Python setup. On Linux, it may be necessary to install it, similarly to `pip`:

```
# install venv like pip, depending on your distribution
sudo apt install python3-venv
```

Then, create the virtual environment, activate it, and install our dependencies into it. Run one step at a time and check each completed successfully before continuing:

```
cd insectai-2026-lightning-session

# create virtual environment
python3 -m venv .venv

# activate - must be done in each new shell
. .venv/bin/activate

# install packages in virtual environment
# on Linux:
pip install -r requirements.txt
# on Mac/Windows:
pip install -r requirements-mac-windows.txt
```

This should install the CPU version of PyTorch. Check the setup with:

```
# check install
python3 src/test-pytorch.py
```

Output should look like this (numbers are random):

```
tensor([[0.2358, 0.4919, 0.7388],
        [0.2353, 0.9614, 0.6958],
        [0.1981, 0.7616, 0.3200],
        [0.8250, 0.9753, 0.4111],
        [0.6613, 0.5657, 0.7407]])

PyTorch CUDA available: False
```

You can now try to run the [PyTorch MNIST example](https://github.com/pytorch/examples/blob/main/mnist/main.py) and make yourself familiar with the code. During the training session we will convert it to PyTorch Lightning.

```
# see available options
python3 src/00-pytorch-example-mnist/main.py --help

# run it!
python3 src/00-pytorch-example-mnist/main.py --save-model
```

## Code Editor / IDE

If you do not yet have a favorite Python code editor or IDE, I recommend giving VS Code a try. It is available for free on Mac, Linux, and Windows: https://code.visualstudio.com/download

For this training session, you will want to install at least the Python, Python Debugger, and Tensorboard extensions in VS Code.

Once installed, open the top-level code directory (i.e. `insectai-2026-lightning-session`) in VS Code. Type Ctrl+Shift+P (or  or ⌘+⇧+P on Mac) to open the search bar, then pick "Python: Select Interpreter". Select the one installed in the virtual environment (`./.venv/bin/python`). 

## CUDA Setup

PyTorch with CPU support will be sufficient for this training session. 

If you want to give CUDA a try (and have the necessary hardware), you will need to have at least a hardware driver, the CUDA toolkit, and PyTorch with CUDA support installed.

Finding the right version CUDA and PyTorch for your hardware can be a bit tricky. Here are some links to guides and instructions that might be helpful.

CUDA compatible GPUs: https://developer.nvidia.com/cuda/gpus

NVIDIA Driver Downloads: https://www.nvidia.com/en-us/drivers/

CUDA Toolkit release notes driver compatibility: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-driver

Older CUDA Toolkit versions: https://developer.nvidia.com/cuda-toolkit-archive

More on CUDA compatibility: https://docs.nvidia.com/deploy/cuda-compatibility/index.html

To install PyTorch with CUDA support using `pip` as described above, see notes in the `requirements.txt` and refer to https://pytorch.org/get-started/locally/ for index selection.
