<#
Before you can run .\pipeline.ps1, you have to set the executionpolicy
for your current process to unrestricted

Set-ExecutionPolicy Unrestricted -Scope Process
#>

# Load environment and install requirements
if (Test-Path -Path ./venv) {
    Write-Output "Loading virtual environment and installing requirements"
    ./venv/Scripts/activate
    # >$null 2>&1 does not show output in terminal
    pip install -r .\requirements.txt >$null 2>&1
} else {
    Write-Output "Creating virtual environment and installing requirements"
    python -m venv venv 
    ./venv/Scripts/activate
    pip install -r .\requirements.txt >$null 2>&1
}

# Run the script
python .\main.py
