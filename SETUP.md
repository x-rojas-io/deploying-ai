# Setup
## Environment Setup Guide
Before using this repo, make sure you’ve completed the [environment setup guide](https://github.com/UofT-DSI/onboarding/blob/main/environment_setup/README.md), which installs the core tools you’ll need for this module, such as:

- Git  
- Git Bash (for Windows)  
- Visual Studio Code
- UV

## Necessary Packages
The Deploying AI module uses its own isolated environment called `deploying-ai-env` so that packages don’t conflict with other projects. 
We use **uv** to create this environment, activate it, and install the required packages listed in the module’s `pyproject.toml`.  
This setup only needs to be done **once per module**, after that, you just activate the environment whenever you want to work in this repo.  

Open a terminal (macOS/Linux) or Git Bash (Windows) in this repo, and run the following commands in order:

1. Create a virtual environment called `deploying-ai-env`:
    ```
    uv venv deploying-ai-env --python 3.11
    ```

2. Activate the environment:
    - for macOS/Linux:
        ```
        source deploying-ai-env/bin/activate
        ```
        
    - for windows (git bash):    
        ```
        source deploying-ai-env/Scripts/activate
        ```

3. Install all required packages from the [pyproject.toml](./pyproject.toml)
    ```bash
    uv sync --active
    ```

> **Note:**  
> - The `deploying-ai-env` folder is intentionally created inside this repository.  
> - This helps Visual Studio Code detect and attach the correct Python kernel automatically.  
> - It’s not hidden because participants should be able to see it, understand its purpose, and, if needed, delete and recreate it.  
> - The folder is safely listed in `.gitignore`, so it won’t be pushed to GitHub.

## Environment Usage
In order to run any code in this repo, you must first activate its environment.
- for macOS/Linux:
    ```
    source deploying-ai-env/bin/activate
    ```
    
- for windows (git bash):    
    ```
    source deploying-ai-env/Scripts/activate
    ```

When the environment is active, your terminal prompt will change to show:  
```
(deploying-ai-env) $
```
This is your **visual cue** that you’re working inside the right environment.  

When you’re finished, you can deactivate it with:  
```bash
deactivate
```

> **👉 Remember**   
> Only one environment can be active at a time. If you switch to a different repo, first deactivate this one (or just close the terminal) and then activate the new repo’s environment.

---

For questions or issues, please contact the Deploying AI Module learning support team or email courses.dsi@utoronto.ca.

## Troubleshooting

### Failed to build `av`

If you encounter an error "Failed to build 'av'" or "You are REQUIRED to use ffmpeg 7", this usually means that a pre-built binary is not available for your system and Python version. To fix this, you must install `ffmpeg` and `pkg-config`.

**macOS (using Homebrew)**:
```bash
brew install ffmpeg pkg-config
```

**Linux (Ubuntu/Debian)**:
```bash
sudo apt-get update && sudo apt-get install -y libavdevice-dev libavfilter-dev libopus-dev libvpx-dev pkg-config
```
*Note: You may need a newer version of ffmpeg (version 7) than what is available in default repositories.*

**Windows**:
Pre-built wheels are usually available. If you encounter this error on Windows, ensure you are using a supported Python version (3.10-3.13) or check the [PyAV documentation](https://pyav.org/docs/stable/overview/installation.html).