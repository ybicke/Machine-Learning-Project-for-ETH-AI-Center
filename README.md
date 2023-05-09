# Machine Learning Project

This is the repository for the AI Center Projects in Machine Learning Research course in 2023 Spring.

## Setting up

### Install dependencies

1. Install [Python 3.11](https://www.python.org/downloads/)
1. Install [Poetry](https://python-poetry.org/docs/#installation)
1. Add `poetry` to the PATH (see step 3 in the [Poetry install instructions](https://python-poetry.org/docs/#installation))
1. Run `poetry install` in the project directory to install dependencies of the project.
   *It might fail to install some dependencies. If it does, install these in the next step, then run `poetry install` again.*
1. Some dependencies are needed to be installed manually. Run `poetry shell`, then `pip install gym==0.21` to install them (it's important that you ran `poetry install` before, so that the virtual environment is created).

### Set up VSCode (recommended)

1. In VSCode, open the command palette with `Ctrl+Shift+P` and choose `Python: Select Interpreter`, then select the virtual environment created by Poetry.
   *Note: If the Poetry environment is not in the list, you can find the location of the environment by running `poetry show -v` and adding a new entry to the list.*
1. Start a new terminal. VSCode will automatically activate the selected environment.
1. Run `pre-commit install` to install pre-commit hooks (they will run some checks before each commit to the repo).
1. Install these VSCode extensions (by searching for them on the extensions tab): `charliermarsh.ruff`, `njpwerner.autodocstring`, `visualstudioexptteam.vscodeintellicode`, `ms-python.black-formatter`, `ms-python.isort`, `ms-python.vscode-pylance`, `ms-python.pylint`, `ms-python.python`, `kevinrose.vsc-python-indent`

### Without VSCode

1. Run `poetry install` in the project directory to install the remaining project dependencies.
1. Run `poetry shell` to activate the virtual environment
1. Run `pre-commit install` to install pre-commit hooks (they will run some checks before each commit to the repo).
1. It's recommended to set up the extensions in your IDE equivalent to those listed above in the VSCode setup section for a more convenient development.

## Managing dependencies

- To add a package to dependencies, use `poetry add <package>`
- To add a package development dependencies (not necessary for running the code), use `poetry add -D <package>`
- To remove a package from the dependencies, use `poetry remove <package>`

## Running the code

You can run scripts specified in `pyproject.toml` with `poetry run <script name>`. For example, to run the `gridworld` script, run `poetry run gridworld`.

To launch the web interface, run:

```bash
poetry run flask --app ml_project/web_interface run
```

(might also need to run `poetry install` to update the dependencies)
