"""
Generates valid, executable Jupyter Notebooks for:
1. 01_eda.ipynb (Exploratory Data Analysis)
2. 02_modeling.ipynb (Model Training, Chronological Validation & Inventory Optimization)
"""

import json
import os

notebooks_dir = "/working_dir/smart-demand-forecasting/notebooks"

# 1. 01_eda.ipynb
eda_nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 01. Exploratory Data Analysis (EDA)\n",
                "## Smart Demand Forecasting Platform\n",
                "\n",
                "This notebook covers:\n",
                "1. Data ingestion of raw retail e-commerce transactions\n",
                "2. Data cleaning (cancellations, negative quantities, missing values, duplicates)\n",
                "3. Daily, weekly, and monthly sales trend analysis\n",
                "4. Product velocity and top-revenue SKUs\n",
                "5. Seasonal and festival surge detection (Black Friday, Christmas rush)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "sns.set_theme(style='whitegrid')\n",
                "raw_path = '../data/raw/ecommerce_transactions.csv'\n",
                "df_raw = pd.read_csv(raw_path)\n",
                "print(f'Raw transactions shape: {df_raw.shape}')\n",
                "df_raw.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Data Cleaning audit\n",
                "from src.data.cleaner import DataCleaner\n",
                "cleaner = DataCleaner(raw_path)\n",
                "df_clean, audit = cleaner.clean()\n",
                "print('Audit Log:')\n",
                "for k, v in audit.items():\n",
                "    print(f'  {k}: {v}')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Revenue by product category\n",
                "cat_rev = df_clean.groupby('Category')['Revenue'].sum().sort_values(ascending=False)\n",
                "print(cat_rev)"
            ]
        }
    ],
    "metadata": {
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

# 2. 02_modeling.ipynb
modeling_nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 02. Modeling, Chronological Validation & Inventory Optimization\n",
                "## Smart Demand Forecasting Platform\n",
                "\n",
                "This notebook covers:\n",
                "1. Chronological Train/Validation/Test split (out-of-time Q4 evaluation)\n",
                "2. Baseline benchmark models (Last Period, 7-Day Moving Average, Seasonal Naive)\n",
                "3. Machine Learning models (Ridge Regression, Gradient Boosted Forecaster)\n",
                "4. Evaluation metrics (MAE, RMSE, sMAPE, WMAPE, Bias)\n",
                "5. Seasonality & festival impact ablation experiment\n",
                "6. Safety stock ($Z \\times \\sigma \\times \\sqrt{L}$) and reorder point determination"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import json\n",
                "\n",
                "from src.models.trainer import DemandModelTrainer\n",
                "trainer = DemandModelTrainer(\n",
                "    features_path='../data/processed/daily_demand_features.csv',\n",
                "    artifacts_dir='../data/processed'\n",
                ")\n",
                "results = trainer.run()\n",
                "pd.DataFrame(results['metrics']).T"
            ]
        }
    ],
    "metadata": {
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(os.path.join(notebooks_dir, "01_eda.ipynb"), "w") as f:
    json.dump(eda_nb, f, indent=2)

with open(os.path.join(notebooks_dir, "02_modeling.ipynb"), "w") as f:
    json.dump(modeling_nb, f, indent=2)

print("Jupyter notebooks generated successfully.")
