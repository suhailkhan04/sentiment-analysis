This project demonstrates a simple sentiment analysis application using Natural Language Processing (NLP) techniques in Python. It classifies given text data into Positive, Negative, or Neutral sentiments using TextBlob for polarity scoring and Multinomial Naive Bayes for model training.

📁 Project Structure
Sentiment Analysis on Text Data using NLP.py — Main Python script for loading data, preprocessing, sentiment labeling, vectorizing, training, and evaluating.

Built-in sample dataset (custom text inputs).

🧠 Key Technologies Used
Python 3

TextBlob (for sentiment polarity)

Pandas, NumPy (for data manipulation)

Matplotlib, Seaborn (for visualization)

Scikit-learn (for model training and evaluation)

📊 Sentiment Labeling Logic
The sentiment is determined based on the polarity score from TextBlob:

Polarity > 0.1 → Positive

Polarity < -0.1 → Negative

Otherwise → Neutral

🚀 How It Works
Sample text data is loaded into a DataFrame.

Sentiment polarity is calculated using TextBlob.

Text is labeled as Positive, Neutral, or Negative.

A count plot shows the distribution of sentiment classes.

CountVectorizer transforms text into numerical features.

Data is split into training and test sets.

A Multinomial Naive Bayes model is trained on the data.

Model is evaluated using a classification report.

📝 Sample Output
markdown
Copy
Edit
               precision    recall  f1-score   support

          -1       1.00      1.00      1.00         1
           0       1.00      1.00      1.00         1

    accuracy                           1.00         2
   macro avg       1.00      1.00      1.00         2
weighted avg       1.00      1.00      1.00         2
Note: Metrics vary depending on train-test split (randomized).

📦 Installation
Ensure Python 3 is installed, then run:

bash
Copy
Edit
pip install pandas numpy seaborn matplotlib textblob scikit-learn
python -m textblob.download_corpora
▶️ Running the Project
bash
Copy
Edit
python "Sentiment Analysis on Text Data using NLP.py"
📌 Future Improvements
Use larger and real-world datasets.

Incorporate deep learning models (e.g., LSTM, BERT).
<img width="1920" height="1080" alt="Screenshot 2025-07-15 113311" src="https://github.com/user-attachments/assets/a38d32b3-7db8-47d6-b224-b16eb7e1f166" />
<img width="1920" height="1080" alt="Screenshot 2025-07-15 113257" src="https://github.com/user-attachments/assets/da1dbd6a-1e66-4b20-bf1f-805adeb482e8" />
<img width="1920" height="1080" alt="Screenshot 2025-07-15 113247" src="https://github.com/user-attachments/assets/afeaeeab-4456-466f-828c-d7febf3024a9" />

Deploy as a web app using Flask or Streamlit.

📃 License
This project is open-source and free to use for educational purposes.
