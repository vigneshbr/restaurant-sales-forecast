# restaurant-sales-forecast

To get a demo of this application on your local server please follow the follwing instructions:

## 1. Requirements:
1. Install 'git' visit and follow instruction to install git: https://git-scm.com/downloads
It is essential to clone this git repository to your local system.

2. conda environment is required to be pre installed.

## 2. Running the app locally

### First create a virtual environment with conda or venv inside a temp folder, then activate it.
* virtualenv venv

#### Windows
* venv\Scripts\activate
#### Or Linux
* source venv/bin/activate

### Clone the git repo, then install the requirements with pip
Run the following commands:

* git clone https://github.com/vigneshbr/restaurant-sales-forecast
* cd restaurant-sales-forecast
* pip install -r requirements.txt

The above commands should be able to clone to this repository and a copy would be available in your local system. Also, all the modules required for this app would be installed.

Furthur, If you face any error like: 
*ModuleNotFoundError: No module named '<module_name>'

Please- 
* pip istall <module_name>
\n This should solve the issue.

### Finally running the app:
run the following command:
python sales.py

If you encounter the following, the app is successfully running on your local server
*Debugger is active!
*Debugger PIN: 271-602-624
*Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

Paste - http://127.0.0.1:5000/ on your web browser, this should be successfully displaying the app.


