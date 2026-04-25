#  Vigilant.AI – Supply Chain Fraud Detection System

A smart, AI-powered system designed to detect fraud in multi-tier supply chain invoices using **Machine Learning, Rule-based logic, and Graph Analysis**.

---

##  Overview

Supply chain fraud is complex and often hidden across multiple entities such as suppliers, buyers, and lenders.
**Vigilant.AI** detects suspicious patterns by combining:

*  Machine Learning (Isolation Forest)
*  Statistical & Rule-based Detection
*  Graph-based Relationship Analysis
*  PDF + CSV Invoice Processing

---

##  Features

###  Fraud Detection Engine

* Detects anomalies using ML (Isolation Forest)
* Identifies duplicate invoices using hashing
* Flags multi-lender usage (same GSTIN across lenders)
* Detects high-value and high-frequency transactions

---

###  Smart Invoice Parsing

* Supports **PDF and CSV uploads**
* Extracts:

  * Invoice ID
  * GSTIN
  * Amount
  * Date
* Handles real-world formats (Flipkart-style invoices included)

---

###  Interactive Dashboard

* KPI Metrics (Total, Fraud, Normal)
* Filtered views (Fraud-only toggle)
* Clean tabular results
* Downloadable CSV output

---

###  Graph Intelligence

* Visualizes supplier → buyer relationships
* Highlights suspicious transaction patterns
* Detects potential circular trading

---

###  Risk Insights

* Supplier-wise risk heatmap
* Top risky invoices ranking
* Data-driven fraud scoring system

---

###  What-if Simulation

* Simulate fraud scenarios by adjusting:

  * Amount
  * Lender count
  * Duplicate flag
* Instantly see risk score changes

---

##  Tech Stack

| Component       | Technology                      |
| --------------- | ------------------------------- |
| Backend         | Python                          |
| ML Model        | Scikit-learn (Isolation Forest) |
| Data Processing | Pandas                          |
| UI              | Streamlit                       |
| Graph Analysis  | NetworkX                        |
| Visualization   | Matplotlib                      |
| PDF Parsing     | pdfplumber                      |

---

##  Project Structure

```
project/
│
├── dashboard.py       # Streamlit UI
├── ml.py              # Fraud detection logic
├── requirements.txt   # Dependencies
└── sample_data/       # Example invoices (optional)
```

---

## ⚙️ Installation & Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/vigilant-ai.git
cd vigilant-ai
```

### 2. Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run dashboard.py
```

---

##  How It Works

1. Upload invoice(s) (PDF/CSV)
2. System extracts structured data
3. ML model detects anomalies
4. Rule-based checks add additional signals
5. Graph analysis detects suspicious relationships
6. Final risk score is computed
7. Results are visualized in dashboard

---

##  Use Cases

*  Invoice financing fraud detection
*  Supply chain monitoring
*  Duplicate invoice prevention
*  GST-based fraud tracking

---

##  Highlights

* Hybrid fraud detection (ML + Rules + Graph)
* Works on real-world invoice formats
* Explainable and interactive system
* Built for hackathon-level innovation

---

##  Limitations

* PDF parsing depends on text clarity
* Not optimized for extremely large datasets (yet)
* Requires structured fields for best performance

---

##  Future Improvements

* NLP-based invoice understanding
* Interactive graph visualization
* Real-time fraud alerts
* API deployment for production

---

##  Contributing

Pull requests are welcome!
For major changes, please open an issue first.

---

##  License

This project is for educational and hackathon purposes.

---

##  Author

Built by **Lokesh Kewat** 🚀
For hackathons and real-world fraud detection innovation.

---


