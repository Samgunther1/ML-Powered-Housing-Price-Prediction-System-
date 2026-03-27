# ML-Powered Housing Price Prediction System

A machine learning project that predicts housing prices using property data scraped from Realtor.com via the HomeHarvest Python package.

---

## Prerequisites

Before you begin, you need to install two things: **Visual Studio Code** and **Python**.

### 1. Install Visual Studio Code

1. Go to [https://code.visualstudio.com](https://code.visualstudio.com)
2. Click the download button for your operating system (Windows, Mac, or Linux)
3. Run the installer
   - **Windows**: Check "Add to PATH" and "Register Code as an editor for supported file types" during installation
   - **Mac**: Drag VS Code into your Applications folder
4. Open VS Code after installation
5. Install the **Python extension**: Click the Extensions icon on the left sidebar (or press `Ctrl+Shift+X`), search for "Python" by Microsoft, and click Install

### 2. Install Python

1. Go to [https://www.python.org/downloads](https://www.python.org/downloads)
2. Download the latest version of Python 3 (3.12 recommended)
3. Run the installer
   - **⚠️ IMPORTANT (Windows):** Check the box that says **"Add Python to PATH"** at the bottom of the first installer screen — do not skip this step
4. Verify the installation by opening a terminal and running:
   ```
   python --version
   ```
   You should see something like `Python 3.12.x`

### 3. Install Git

1. Go to [https://git-scm.com/downloads](https://git-scm.com/downloads)
2. Download and run the installer for your OS
   - **Windows**: The default settings during installation are fine
3. Verify by running:
   ```
   git --version
   ```

---

## Getting Started

### Step 1: Clone the Repository

Open a terminal (or the VS Code integrated terminal) and run:

```bash
git clone https://github.com/YOUR-USERNAME/ML-Powered-Housing-Price-Prediction-System.git
cd ML-Powered-Housing-Price-Prediction-System
```

### Step 2: Open the Project in VS Code

From inside the project folder, run:

```bash
code .
```

Or open VS Code manually and go to **File → Open Folder**, then navigate to the project folder.

### Step 3: Create a Python Virtual Environment

Open the integrated terminal in VS Code (`Ctrl + ~`) and run:

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

> **If you get an error about execution policy**, run this command first, then try activating again:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**Mac / Linux:**

```bash
source .venv/bin/activate
```

You should see `(.venv)` at the beginning of your terminal prompt when the environment is active.

### Step 5: Select the Python Interpreter in VS Code

1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type **"Python: Select Interpreter"** and select it
3. Choose the interpreter that points to `.venv` (it will say something like `.venv/Scripts/python.exe` or `.venv/bin/python`)

### Step 6: Install Dependencies

With the virtual environment activated, run:

```bash
pip install -r requirements.txt
```

This installs all the Python packages the project uses.

---

## Basic Git Workflow

Every time you sit down to work on the project, follow this pattern:

```bash
# 1. Pull the latest changes from GitHub
git pull origin main

# 2. Do your work...

# 3. Stage your changes
git add .

# 4. Commit with a descriptive message
git commit -m "Added feature engineering for sqft and lot size"

# 5. Push your changes to GitHub
git push origin main
```

### Tips to Avoid Merge Conflicts

- **Always pull before you start working**
- Communicate with the team about which files you are editing
- Commit and push frequently — small commits are better than large ones

---

## Project Structure

```
ML-Powered-Housing-Price-Prediction-System/
├── data/                  # Raw and processed datasets
├── notebooks/             # Jupyter notebooks for EDA and modeling
├── src/                   # Source code and scripts
├── models/                # Saved model artifacts
├── requirements.txt       # Python package dependencies
├── .gitignore             # Files/folders excluded from Git
└── README.md              # This file
```

---

## Updating Dependencies

If you install a new package for the project, update `requirements.txt` so the rest of the team gets it too:

```bash
pip install some-new-package
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Added some-new-package to requirements"
git push origin main
```

Your teammates then just need to pull and run `pip install -r requirements.txt` again.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `python` is not recognized | Python wasn't added to PATH — reinstall and check the "Add to PATH" box |
| `git` is not recognized | Git wasn't added to PATH — reinstall Git |
| PowerShell won't activate the venv | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `pip install` fails with permission errors | Make sure your virtual environment is activated (you should see `(.venv)` in your prompt) |
| Merge conflict on `git pull` | Open the conflicted file, look for `<<<<<<<` markers, resolve manually, then commit |
