# Financial Transactions Analysis

This repository contains a Python-based data analysis and machine learning project. It uses a dataset of financial transactions to train a linear regression model.

## Project Description

The project involves preprocessing a dataset, training a linear regression model using the preprocessed data, and evaluating the model's performance. The dataset contains financial transactions with various features like date, name, income, expenditure, and others.

## Installation

Clone the repository to your local machine and install the necessary Python packages using pip:

```bash
git clone https://github.com/stephenWanjala/APAbreakdown.git
cd APAbreakdown
pip install -r requirements.txt
```

## Usage

Run the Python script in your preferred IDE or from the command line:

```bash
jupiter notebook model.ipynb
```

## Results

The model's performance is evaluated using Mean Squared Error (MSE) and R-squared (R2) score. The results are printed to the console.

## Model Persistence

The trained model is saved to a file named `predictions_model.pkl` using joblib.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the [MIT License](LICENSE).