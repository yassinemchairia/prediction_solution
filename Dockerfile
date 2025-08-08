# Use a lightweight Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Create logs directory
RUN mkdir -p logs

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY solution_prediction_api.py .
COPY solution_model.pkl .
COPY tfidf_vectorizer.pkl .
COPY alertes.csv .

# Expose port 5000
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "solution_prediction_api.py"]
