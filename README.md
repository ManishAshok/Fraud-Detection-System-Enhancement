<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>Fraud Detection System Enhancement</h1>

<h2>Project Overview</h2>
<p>This project focuses on enhancing fraud detection capabilities by analyzing transaction data to identify suspicious activities. The solution leverages machine learning models implemented in Python and R, with deployment on a Hadoop framework. SQL is used for data querying and preprocessing.</p>

<h3>Technologies Used:</h3>
<ul>
    <li><strong>Hadoop:</strong> For distributed data storage and processing.</li>
    <li><strong>SQL:</strong> For data extraction and preprocessing.</li>
    <li><strong>Python:</strong> For data preprocessing, feature engineering, model training, and evaluation.</li>
    <li><strong>R:</strong> For additional analysis and visualization (optional).</li>
</ul>

<h2>Project Structure</h2>

<h3>1. Data Files</h3>
<ul>
    <li><strong>transaction_data.csv:</strong> Synthetic transaction data including transaction amount, location, and fraud status.</li>
    <li><strong>X_train.csv, y_train.csv:</strong> Training set for model training.</li>
    <li><strong>X_test.csv, y_test.csv:</strong> Test set for model evaluation.</li>
</ul>

<h3>2. Python Scripts</h3>
<ul>
    <li><strong>data_preprocessing.py:</strong> Preprocesses the transaction data, normalizes features, and splits the data into training and testing sets.</li>
    <li><strong>model_training.py:</strong> Trains a Random Forest classifier on the preprocessed data and evaluates its performance.</li>
    <li><strong>eda.py:</strong> Performs exploratory data analysis (EDA) on the transaction data.</li>
    <li><strong>confusion_matrix.py:</strong> Generates a confusion matrix to evaluate the model's performance.</li>
</ul>

<h3>3. R Script (Optional)</h3>
<ul>
    <li><strong>eda.R:</strong> Additional data visualization using R (if you prefer to use R for EDA).</li>
</ul>

<h2>How to Execute the Project</h2>

<h3>1. Clone the Repository</h3>
<p>First, clone this repository to your local machine:</p>
<pre><code>git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system</code></pre>

<h3>2. Install Dependencies</h3>
<p>Ensure you have Python installed, along with the necessary packages:</p>
<pre><code>pip install pandas numpy scikit-learn matplotlib seaborn</code></pre>

<h3>3. Data Preprocessing</h3>
<p>Run the <code>data_preprocessing.py</code> script to preprocess the transaction data:</p>
<pre><code>python data_preprocessing.py</code></pre>

<h3>4. Exploratory Data Analysis</h3>
<p>You can perform EDA using Python:</p>
<pre><code>python eda.py</code></pre>
<p>Alternatively, if you prefer R for EDA, use:</p>
<pre><code>Rscript eda.R</code></pre>

<h3>5. Model Training and Evaluation</h3>
<p>Train and evaluate the model by running the <code>model_training.py</code> script:</p>
<pre><code>python model_training.py</code></pre>

<h3>6. Generate Confusion Matrix</h3>
<p>After training the model, you can generate a confusion matrix to evaluate its performance:</p>
<pre><code>python confusion_matrix.py</code></pre>

<h3>7. Integrating Hadoop</h3>
<p>To integrate Hadoop for distributed processing, follow these steps:</p>
<ol>
    <li><strong>Set Up Hadoop Cluster:</strong>
        <ul>
            <li>Set up a Hadoop cluster (local or cloud-based) and ensure HDFS is running.</li>
            <li>Place the preprocessed data into HDFS:</li>
        </ul>
        <pre><code>hdfs dfs -put X_train.csv /user/yourusername/fraud-detection/X_train.csv
hdfs dfs -put y_train.csv /user/yourusername/fraud-detection/y_train.csv</code></pre>
    </li>
    <li><strong>Modify the Python Scripts:</strong>
        <p>Adjust the scripts to load data from HDFS using PySpark instead of pandas.</p>
        <pre><code>from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder.appName("FraudDetection").getOrCreate()

# Load the data from HDFS
data = spark.read.csv("hdfs:///user/yourusername/fraud-detection/X_train.csv", header=True, inferSchema=True)

# Assemble features into a vector
assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol="features")
data = assembler.transform(data)

# Train the model
rf = RandomForestClassifier(labelCol="is_fraud", featuresCol="features", numTrees=100)
model = rf.fit(data)

# Evaluate the model on the test data
predictions = model.transform(data)
evaluator = MulticlassClassificationEvaluator(labelCol="is_fraud", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Model Accuracy: {accuracy:.2f}")</code></pre>
    </li>
    <li><strong>Run the Job on Hadoop:</strong>
        <p>Submit the job to your Hadoop cluster using <code>spark-submit</code>:</p>
        <pre><code>spark-submit --master yarn model_training.py</code></pre>
    </li>
</ol>

<h3>8. Viewing Results</h3>
<p>After running the scripts, check the output files for model performance metrics, including the confusion matrix, accuracy, precision, recall, and F1-score.</p>

<h2>Contributing</h2>
<p>Feel free to submit a pull request or open an issue if you have suggestions or improvements.</p>

<h2>License</h2>
<p>This project is licensed under the MIT License.</p>

</body>
</html>
