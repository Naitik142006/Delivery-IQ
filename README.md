# 🚀 DeliveryIQ: Smart Food Delivery Analytics & Prediction Platform

DeliveryIQ is a high-performance, production-ready analytics dashboard built with **Python** and **Streamlit**. It empowers logistics managers and data analysts to visualize delivery performance, detect operational bottlenecks, and predict delivery durations using machine learning.

![Premium UI](https://img.shields.io/badge/UI-Premium_Dark_Mode-blueviolet?style=for-the-badge)
![ML Powered](https://img.shields.io/badge/ML-Random_Forest-emerald?style=for-the-badge)
![Framework](https://img.shields.io/badge/Framework-Streamlit-ff4b4b?style=for-the-badge)

---

## ✨ Key Features

### 🏢 Manager Dashboard
Designed for high-level operational oversight:
- **Decision Score**: A dynamic 0-100 efficiency rating based on real-time delivery latency and satisfying metrics.
- **Intelligent Recommendations**: AI-driven suggestions for kitchen staffing, delivery re-zoning, and VIP fast-tracking.
- **Geographic Intelligence**: Interactive map visualizations showing delivery hotspots, revenue density, and latency heatmaps.
- **System Alerts**: Auto-fading notifications for critical delays or peak demand surges.

### 📈 Professional Analyst Mode
Deep-dive into the data with granular controls:
- **Interactive Visualizations**: Dynamic Plotly charts for order volume trends, revenue distribution, and delivery time histograms.
- **SLA Analysis**: Identify on-time vs. late delivery ratios and their impact on customer satisfaction.
- **Top Performers & Laggards**: Automated identification of the slowest deliveries for root-cause analysis.

### 🔮 Delivery Prediction Simulator
Built-in machine learning engine:
- **What-If Analysis**: Use sliders to simulate delivery scenarios (changing distance, time of day) and see predicted durations instantly.
- **Random Forest Engine**: Predicts delivery times based on historical training data included in the platform.

### 🎨 Premium User Experience
- **Theme Switcher**: Toggle between **Modern Neon** and **Business Blue** aesthetics.
- **Responsive Layout**: Optimized for various screen sizes with custom glassmorphism effects.
- **Auto-Cleaning**: Intelligent data health check that automatically maps columns and imputes missing values.

---

## 🛠️ Tech Stack

- **Frontend/Dashboard**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (RandomForestRegressor)
- **Data Engineering**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Visualization**: [Plotly Express](https://plotly.com/python/)
- **Styling**: Vanilla CSS (Custom Glassmorphism & Animations)

---

## 🏃 Setup & Installation

### 1. Prerequisites
- Python 3.9+
- Virtual environment (recommended)

### 2. Clone and Install
```bash
# Install dependencies
pip install -r requirements.txt
```

### 3. (Optional) Re-train the Model
The project comes with a pre-trained `model.pkl`. To re-train it on your own data:
```bash
python train_model.py
```

### 4. Launch the Platform
```bash
streamlit run app.py
```

---

## 📂 Project Structure

- `app.py`: The main multi-mode Streamlit application.
- `style.css`: Custom CSS for the premium dark-mode UI.
- `train_model.py`: Script to generate the Random Forest model.
- `food_delivery_cleaned.csv`: Sample dataset for demonstration.
- `requirements.txt`: Required Python packages.

---

## ☁️ Deployment
DeliveryIQ is fully compatible with **Streamlit Cloud**:
1. Upload this repository to GitHub.
2. Connect your repository to Streamlit Cloud.
3. Ensure `requirements.txt` and `model.pkl` are in the root directory.
4. Set `app.py` as the entry point.

---
*Developed with ❤️ for the future of Smart Logistics.*

