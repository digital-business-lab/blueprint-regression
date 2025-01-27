# blueprint-regression
This repository holds code for a regression-task pipeline. You can configure your settings for the model / dataset / preprocessing in ./config/config.yaml.

# Table of Content
- [Setup](#setup)
    - [1. Setup](#1-setup-pipelineps1)
    - [2. Setup](#2-setup-manually)
- [Troubleshooting](#troubleshooting)
- [Contact](#contact)


# Setup
## 1. Setup (pipeline.ps1)
If you do not want to set everything up manually just execute the file ./pipeline.ps1.
1. Configure [Config](./config/config.yaml) for your requirements. An explanation of all the variables is [here](./config/README.md).
2. To run powershell scripts on your PC you have to set ExecutionPolicy to Unrestricted. In your terminal type:
```powershell
Set-ExecutionPolicy Unrestricted -Scope Process
```
3. Execute ./pipeline.ps1 (Before executing look at the code and understand it)
```powershell
./pipeline.ps1
```

## 2. Setup (manually)
With this method you can setup the repo step by step by yourself.
1. Configure [Config](./config/config.yaml) for your requirements. An explanation of all the variables is [here](./config/README.md).

2. To activate the virtual environment on your PC you have to set ExecutionPolicy to Unrestricted. In your terminal type:
```powershell
Set-ExecutionPolicy Unrestricted -Scope Process
```

3. Create or activate an existing virtual environment
```powershell
python -m venv venv
./venv/Scripts/activate
```

4. Install the requirements in your virtual environment
```powershell
pip install -r requirements.txt
```

# Troubleshooting
If you have any questions regarding the code, algorithm, or any related issues feel free to submit a problem report to the specified email provided in the [contacts](#contact) section. 
To help us assist you efficiently, please ensure your problem report is well-structured and includes the following details:

1. **Environment Details**: Specify your operating system (OS), Python version and any relevant software or library versions
2. **Summary**: Provide a concise summary of the issue you're facing and the goal you are trying to achieve  
3. **Problem Statement**: Clearly describe the problem, including any error messages, unexpected behavior, or obstacles
4. **Reproducibility Steps**: Outline the steps to reproduce the issue. Include any code snippets, datasets (if applicable) or configurations that are essential for recreating the problem

Providing this information ensures we can diagnose and resolve your issue as quickly and effectively as possible.

# Contact
If you have questions about this repo you can contact lukas.graf.zangenstein@gmail.com.
