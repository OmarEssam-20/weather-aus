# WeatherAUS EDA Streamlit App

This repository contains an **interactive Streamlit web application** for performing  
**Exploratory Data Analysis (EDA)** on the WeatherAUS dataset.

The app includes:

- Data preview  
- Missing values analysis  
- Interactive distributions  
- Correlation heatmap  
- Target (RainTomorrow) analysis  
- Location-based rainfall probability  
- Sidebar filters (Location, Season, Target)  
- Automatic preprocessing similar to the Jupyter notebook version  

---

## 📂 Project Structure

```
weather_app.py        # Main Streamlit application
weatherAUS.csv        # Dataset used for EDA
requirements.txt      # Required Python dependencies
README.md             # Project documentation
```

---

## 🚀 Running the App Locally

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Streamlit app
```bash
streamlit run weather_app.py
```

The app will start at:
```
http://localhost:8501
```

---

## 🌐 Deployment Options

You can deploy this Streamlit app using:

### **1. Streamlit Cloud** (recommended)
- Connect your GitHub repo  
- Select `weather_app.py`  
- Deploy — automatic free hosting  

### **2. HuggingFace Spaces**
- Create a new Space  
- Choose **Streamlit**  
- Upload files  
- Public app link is generated instantly  

### **3. Render.com**
- Deploy as a web service  
- Set correct start command  

---

## 🧰 Requirements

```
streamlit
pandas
numpy
matplotlib
seaborn
plotly
```

---

## ✨ Features

- Clean, interactive and easy-to-use UI
- Fully reproducible data cleaning + EDA pipeline
- Works directly on the WeatherAUS public dataset
- Great for data analytics presentations or ML pipeline preparation

---

## 👤 Author

Developed by **Omar Essam** with support from AI tools.

---

## 📬 Contact

If you'd like help deploying or extending the app (adding ML models, predictions, dashboards), feel free to reach out.

Enjoy exploring the weather data! 🌦️
