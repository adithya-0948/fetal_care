# fetal_care
# *Project README – Fetal Abnormalities Prediction System*

## *1. Project Overview*

This project is a machine learning-based system that predicts fetal health conditions using cardiotocography (CTG) data. The system classifies fetal conditions into *Normal, Suspect, and Pathological* categories, helping healthcare professionals in early detection and decision-making.

---

## *2. System Requirements*

### *Hardware Requirements*

* Processor: Intel i3 or above
* RAM: Minimum 4 GB (8 GB recommended)
* Storage: 256 GB HDD/SSD
* Internet Connection (optional for setup)

### *Software Requirements*

* Operating System: Windows / Linux / macOS
* Programming Language: Python (3.7 or above)
* Tools: Jupyter Notebook / VS Code / PyCharm

---

## *3. Installation Guidelines*

### *Step 1: Install Python*

Download and install Python from:
[https://www.python.org/downloads/](https://www.python.org/downloads/)

---

### *Step 2: Install Required Libraries*

Open Command Prompt / Terminal and run:

bash
pip install numpy pandas matplotlib seaborn scikit-learn django


---

### *Step 3: Install Anaconda (Optional)*

Download from:
[https://www.anaconda.com/](https://www.anaconda.com/)

This helps in managing environments and running Jupyter Notebook easily.

---

## *4. Project Setup*

### *Step 1: Download/Clone the Project*

bash
git clone <your-project-link>
cd fetal-health-prediction


### *Step 2: Open Project*

* Open in *VS Code / PyCharm / Jupyter Notebook*

---

## *5. Execution Steps*

### *Step 1: Run Data Preprocessing*

bash
python preprocessing.py


### *Step 2: Train the Model*

bash
python train_model.py


### *Step 3: Run Prediction*

bash
python predict.py


---

### *(For Django Web Application)*

### *Step 4: Run Server*

bash
python manage.py runserver


### *Step 5: Open in Browser*

Go to:


http://127.0.0.1:8000/


---

## *6. How to Use the System*

1. Open the web application
2. Login (if authentication is enabled)
3. Enter CTG input values (FHR, UC, etc.)
4. Click on *Predict*
5. View result:

   * Normal
   * Suspect
   * Pathological

---

## *7. Output*

* Displays predicted fetal condition
* Provides basic interpretation of result
* Helps in clinical decision support

---

## *8. Troubleshooting*

* Ensure all libraries are installed correctly
* Check Python version compatibility
* Verify dataset file path
* Restart server if Django fails

---

## *9. Future Enhancements*

* Integration with real-time hospital systems
* Use of deep learning models
* Mobile application development
* Cloud deployment

---

## *10. Author*

* Name: Adithya
* Muktha
* vinay
* Manasa
* Project: Fetal Abnormalities Prediction System
