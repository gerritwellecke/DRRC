# Contributing to DRRC
PRs are welcome any time!
If you'd like a more general instruction aimed at beginners, keep reading below.

## Automated Documentation
This project comes with a Sphinx documentation for all "package" code, i.e. everything that is in [Src](/Src/).

If you adhere to *Google style* docstrings, then sphinx and napoleon will generate an automated documentation.
There is a pipeline in place that automatically updates after every push / merge to `main` (keep in mind, that the pipeline needs to successfully run once, so it may take a few minutes until a new version is available).

There is a test in place that exposes the current state of the online documentation in every merge request. 
You can find it when clicking on the respective job in the MR pipeline.
The READMEs in this repo will only contain developer information. 
All other documentation will be migrated to the online version for usability in due time.


## First steps to work in this repository
1. Make sure your user name and email address are set correctly on your machine. 
   Execute `git config --global -e` and add the following to the configuration
   ```gitconfig
   [user]
       name = "First Last"
       email = first.last@ds.mpg.de
   ```
   Just use your name and a working email address.

2. I recommend you set up an `ssh` keypair for easy access to this repository.


## Working in this repository 
The `main` branch is protected!
This means that no user is allowed to push directly to main. 
The proposed workflow is as follows:
1. Create a new branch locally using `git checkout -b NEW-BRANCH-NAME`.
2. Then make your changes and commit them. 
   Regarding commit messages, think along these lines:
   - The first word should be a verb in its infinitive form.
     E.g.: [Add gitignore and some description in README]
   - The commit message should have a title of no more than 72 characters, followed by an empty line. 
     Thereafter more details can be written in a body that should be no wider than 80 characters. 
     This vastly improves readability of the git log, but does take some practice. 
     **No worries, these are guidelines, not laws -- as long as it's clear what the commit does you can deviate from this.**
3. Push this new branch upstream. 
   Usually it's sufficient to execute `git push` and git will tell you the right command to create the upstream branch. 
4. Go to the web interface and make a merge request. 
5. After review of whatever changes you made your branch can be merged to `main`.
   It is then best practice to delete the new branch remotely (the web interface lets you do that during the merge) as well as locally (use `git branch -d NEW-BRANCH-NAME`). 
   This prevents that the same history is later submitted to `main` again. 
   To get all changed branches, switch back to main & update the same, and finally clean up the branches one can execute the following:
   ```bash
   git fetch --all
   git checkout main 
   git pull
   git branch -d NEW-BRANCH-NAME 
   git fetch --prune origin
   ```
6. For the next change after a merge, make a new branch, as in 1.


### Avoiding "it works on my machine"-like problems
Upon first use create a virtual environment. 
You can do this in multiple other ways, e.g. with `conda`, but I prefer the built-in method python provides:
```bash
python3 -m venv .venv
```
This will create a virtual environment in the current directory (should be `/Src/`) with the name `.venv`, i.e. the full path will be `/Src/.venv/`.
You then enter the virtual environment with
```bash
source .venv/bin/activate
```
Now your prompt should reflect this environment.
From here you install the packages defined in `requirements.txt` with
```bash
pip install -r requirements.txt
```
You can also use pip as usual to install new packages.
To overwrite the current state of packages afterwards execute
```bash
pip freeze > requirements.txt
```
Then push the new `requirements.txt` to the git repository.

To leave the virtual environment simply use
```bash
deactivate
```
This works from any current working directory.

### Formatting Code
In order to ensure good style in our `python` code, we commit to using `black` and `isort`.
Before any Merge Request can be granted the code must be formatted using
```bash
black Src/
isort --profile black Src/
```
Alternatively one can use the script `format_code.sh` from the repository's root.

If you want to **automate code formatting**, you can use the script `create_git_hook.sh` which will set up a git pre-commit hook, so that the above commands are executed before each commit.
This is, however, **not recommended** as changing files in a pre-commit hook can introduce breaking changes that then are committed without further review. 

A better approach is to append the `--check` flag to both `black` and `isort`, so that the pre-commit hook only checks for code complicance and rejects the commit, if code is unformatted. 

Generally, the better place for formatting automation is in your *editor*!
`Vim` has plugins for that. 
`vscode` does as well. 
I suggest making use of these features. 

There is a pipeline in place, that will demand that all further merge requests agree with the above code formatting standards.

