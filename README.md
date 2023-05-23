# Machine Learning Project

This is the repository for the AI Center Projects in Machine Learning Research course in 2023 Spring.

## Setting up

### Install dependencies

1. Install [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html)
1. Navigate to the project directory, then run `conda create --name ml_project --file conda-<linux/osx/win>-64.lock` (replace `<linux/osx/win>` with your OS) to create the Anaconda environment.
1. Run `conda activate ml_project` to activate the virtual environment.
1. Run `poetry install` in the project directory to install dependencies of the project.
1. To install MuJoCo, follow the [instructions in the GitHub repo](https://github.com/openai/mujoco-py/blob/9ea9bb000d6b8551b99f9aa440862e0c7f7b4191/README.md#install-mujoco).
1. Add the `<user directory>\.mujoco\mjpro150\bin` folder (replace `<user directory>` with your user's directory name) to your PATH.
1. Try installing `mujoco-py` by running `pip install mujoco-py==1.50.1.68`. If it succeeds, jump to [Test MuJoCo](#test-mujoco). If this fails, and your OS is Windows, proceed to [Install MuJoCo on Windows](#install-mujoco-on-windows) section.

#### Install MuJoCo on Windows

1. Install the C++ workload for Visual Studio 2019 in one of the following ways:
   - If you already have Visual Studio 2019 or earlier installed, make sure the C++ build workload is also installed
   - Otherwise either download and install the latest "Build Tools for Visual Studio 2019" from the [Visual Studio downloads](https://my.visualstudio.com/Downloads?q=Visual%20Studio%202019), then select "Desktop development with C++" from the workloads to install
   - Or install the following Chocolatey packages: [Visual Studio 2019 build tools](https://community.chocolatey.org/packages/visualstudio2019buildtools), [Visual C++ 2019 build tools](https://community.chocolatey.org/packages/visualstudio2019-workload-vctools) (in this order)
1. Open "x64 Native Tools Command Prompt for VS 2019" (in `C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Visual Studio 2019\Visual Studio Tools\VC`)
1. Run `conda activate ml_project` to activate the Anaconda environment
1. Try `pip install mujoco-py==1.50.1.68`. If it succeeds, proceed to the [Test MuJoCo](#test-mujoco) section.
1. If the above command fails with "file name too long" error, use `git clone https://github.com/openai/mujoco-py` to clone the repository (clone it to a path that's not very long), then `git checkout a9f563cbb81d45f2379c6bcc4a4cd73fac09c4be` to check out the `1.50.1.68` version of the package.
1. Still inside the x64 Native Tools Command Prompt and the Anaconda environment activated, navigate inside the cloned repository, then run `pip install -r requirements.txt` and `pip install -r requirements.dev.txt` to install the requirements.
1. Run `python setup.py install` to install `mujoco-py` package from source.

#### Test MuJoCo

*Note: If you installed `mujoco-py` from source, you will also need to do the following steps in the x64 Native Tools Command Prompt.*

1. Run `python` to start the python interpreter.
1. If you are on Windows, run `import os; os.add_dll_directory("C:\\Users\\<username>\\.mujoco\\mjpro150\\bin")` (replace `<username>` with your user's folder name). You will need to do this every time before importing `mujoco_py`, otherwise it will detect an import error and try to rebuild itself.
1. Run `import mujoco_py` to import the library. This will also build the library if it hasn't been built before.

If these steps succeed without errors, then the library is successfully installed.

### Set up VSCode (recommended)

1. In VSCode, open the command palette with `Ctrl+Shift+P` and choose `Python: Select Interpreter`, then select the virtual environment created by Anaconda.
   *Note: If the desired environment is not in the list, you can find the location of the environments by running `conda env list` and adding a new entry to the list.*
1. Start a new terminal. VSCode will automatically activate the selected environment.
1. Run `pre-commit install` to install pre-commit hooks (they will run some checks before each commit to the repo).
1. Install these VSCode extensions (by searching for them on the extensions tab): `charliermarsh.ruff`, `njpwerner.autodocstring`, `visualstudioexptteam.vscodeintellicode`, `ms-python.black-formatter`, `ms-python.isort`, `ms-python.vscode-pylance`, `ms-python.pylint`, `ms-python.python`, `kevinrose.vsc-python-indent`

*Note: for now VSCode does not use the selected interpreter for git commands (see [issue](https://github.com/microsoft/vscode-python/issues/10165)), so you need to create commits from the command line.*

### Without VSCode

1. Run `conda activate ml_project` to activate the Anaconda environment.
1. Run `pre-commit install` to install pre-commit hooks (they will run some checks before each commit to the repo).
1. It's recommended to set up the extensions in your IDE equivalent to those listed above in the VSCode setup section for a more convenient development.

## Managing dependencies

- To **add** a package to dependencies, use `poetry add <package>` for dependencies or `poetry add -D <package>` for development dependencies
- If adding a package using Poetry fails (or if a version from Anaconda is needed), add the package and its version to `environment.yml`, then run:
  - `conda-lock -k explicit --conda mamba` to update the Anaconda lock file
  - `mamba update --file conda-<linux/osx/win>-64.lock` (replace `<linux/osx/win>` with your OS) to update the packages in your current environment
- To **remove** a package from the dependencies, use `poetry remove <package>` or remove them from `environment.yml`. In the latter case, you will also need to run the commands above to update the Anaconda lock files and current environment packages.

## Running the code

You can run scripts specified in `pyproject.toml` with `poetry run <script name>`. For example, to run the `gridworld` script, run `poetry run gridworld`.

To launch the web interface, run:

```bash
poetry run flask --app ml_project/web_interface run
```

(might also need to run `poetry install` to update the dependencies)
