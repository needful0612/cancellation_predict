# Hotel Cancellation Prediction Service

This project aims to predict the probability of reservation cancellations. It is designed to help hotels manage revenue by identifying high-risk bookings before they happen, allowing them to provide targeted discounts or other perks to improve retention.

## 1. Problem Description
In the hospitality industry, **cancellations** lead to significant revenue loss and operational inefficiency. This project builds a **binary classification model** to address this.

**Goal:** Predict if a customer will cancel (`is_cancelled = 1`) or honor their booking (`is_cancelled = 0`). High-risk bookings can be flagged for:
* Manual follow-up or confirmation calls.
* Strategic overbooking to mitigate loss.
* Offering retention incentives like further discounts or perks.

## 2. Data Source
The data is generated using a custom script found in `src/generate_mock_data.py`. 
* **Design:** The script features real-world distributions and observed patterns (e.g., the relationship between lead time and cancellation risk).
* **Privacy:** No real customer data is used due to privacy concerns; however, the synthetic data is designed to mimic actual business constraints and behavior.

you can run the script in src folder to see the output of dataset.


## 3. Exploratory Data Analysis (EDA)
The EDA is documented in `notebooks/eda.ipynb`. The primary focus was validating the integrity of the synthetic dataset and preparing it for modeling:

* **Data Quality Check:** Verified that the synthetic generation produced the expected distributions for features like `total_price` and `lead_time`.
* **Target Balance:** Confirmed the distribution of the target variable (`is_cancelled`) to ensure the model wouldn't be overly biased toward a single class.
* **Feature Distributions:** Identified that numerical features like `total_price` were skewed, justifying the use of a **Log Transformation** in the final pipeline.
* **Categorical Cardinality:** Observed high cardinality in the `hotel_id` field, which led to the decision to use **Target Encoding** rather than standard One-Hot Encoding.

## 4. Model Evaluation & Training
The evaluation process is documented in `notebooks/model_evaluation.ipynb`. A **Scikit-Learn Pipeline** was implemented to handle all preprocessing and modeling in a single object.

* **Models Tested:** Logistic Regression, Random Forest Classifier (Selected), and XGBoost.
* **Techniques Used:**
    * **Target Encoding:** For high-cardinality categorical data (`hotel_id`).
    * **Robust Scaling:** For numerical features with outliers.
    * **Log Transformation:** For skewed price data.
    * **One-Hot Encoding:** For categorical data like `payment_type`.
* **Hyperparameter Tuning:** Randomized search was used to fine-tune the Random Forest parameters.
* **Threshold Optimization:** The decision threshold was set to **0.39** to balance Precision and Recall for business needs.

## 5. Dependency and Environment Management
This project uses `pip` and a `requirements.txt` file for environment management.

### Prerequisites
* Python (suggested version: 3.13)
* Docker and Docker Compose

### Local Setup
```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## 6. Reproducibility & Containerization
The project is containerized using **Docker** and orchestrated with **Docker Compose** to ensure the environment is reproducible across different machines.

### Service Architecture
1. **Trainer Service:** Automatically generates mock data and trains the model. It saves the final artifact (`final_artifact.joblib`) to a shared volume.
2. **Serving Service:** A lightweight FastAPI application that loads the trained model from the shared volume and serves predictions via a REST API.
3. **Shared Volume:** A volume bridge (`./models`) allows the Trainer to pass the model to the Server without manual file movement or rebuilding images.

### Reproducing this Project

#### Option 1: Docker Compose (Recommended)
To build and run the entire pipeline (Data Gen -> Train -> Serve) with a single command:
```bash
docker-compose up --build
```
you should see output like below:
```bash
trainer-1  | --- Results ---
trainer-1  | Best Val Threshold: 0.34
trainer-1  | Final Test AUC:      0.8905
trainer-1  | Artifact saved to models/final_artifact.joblib
trainer-1 exited with code 0
server-1   | INFO:     Started server process [1]
server-1   | INFO:     Waiting for application startup.
server-1   | INFO:     Application startup complete.
server-1   | INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```
### Option 2: Running Locally (Notebooks & Scripts)

If you wish to explore the notebooks or run the scripts without Docker, follow these steps to ensure file paths remain consistent across the project:

**Generate Mock Data**(Run the data generation script from the **project root directory**):
```bash
python src/generate_mock_data.py
```
**notebooks**(same directory as above):
```bash
jupyter notebook
```
If running individual Python scripts, ensure your `DATA_PATH` and `MODEL_OUTPUT_PATH` constants match your local directory structure.

## 7. Model Serving (Usage)
The API exposes a POST endpoint at /predict. It validates incoming data using Pydantic to ensure all required fields are present and format the data. to test it run the curl below.

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "brand_id":1,
    "hotel_id":10,
    "room_qty":2,
    "total_price":1200.0,
    "prepaid":200.0,
    "payment_type":"Credit Card",
    "order_date":"2024-01-01",
    "checkin_date":"2024-01-15",
    "customer_order_count":3,
    "checkout_date":"2024-01-20"
}'
```
the output should be something like this:
```bash
{"cancel prob":0.5229,"predict":"Cancelled","is_cancelled":true}
```

## 8. Cloud Deployment

The service is designed to be cloud-agnostic using Docker. To deploy this to a production environment like **AWS Elastic Beanstalk**:

1. **Build locally:** Run the trainer to ensure `models/final_artifact.joblib` is generated.
2. **Initialize EB:** Run `eb init -p docker hotel-service`.
3. **Deploy:** Run `eb create hotel-env`.

The API would then be accessible via a public AWS load balancer URL, allowing integration with hotel management dashboards or mobile apps.

## 9. Cleanup
To stop the services and remove the containers, networks, and shared volumes, use:
```bash
docker-compose down -v --rmi all
```

## 10. Conclusion

During the model evaluation phase, hyperparameter fine-tuning yielded marginal improvements over the baseline configuration. Consequently, the **original parameters** were retained to maintain model simplicity and avoid over-fitting, as the performance gains from tuning did not justify the increased complexity.

Key Takeaways:
* **Model Selection:** While Random Forest, XGBoost, and Logistic Regression all showed viable performance, **Random Forest** was selected as the final model due to its superior AUC-ROC score and stability across cross-validation folds.
* **Trade-offs:** The other models remains viable alternatives; however, the Random Forest pipeline provided the most reliable probability estimates for the 0.39 decision threshold.
* **Impact:** By shifting the focus from global accuracy to threshold-based recall, the service is better equipped to identify high-risk cancellations that would otherwise be missed by a standard 0.5 default.