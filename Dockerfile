FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install pipenv
RUN pip install pipenv

# Copy Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock ./

# Install dependencies using pipenv
# --system flag installs packages into the system python instead of creating a virtual environment
RUN pipenv install --system --deploy

# Copy files maintaining directory structure
ADD model /app/model
COPY train.py /app/train.py
COPY predict.py /app/predict.py
COPY model.py /app/model.py

COPY winequality.csv /app/winequality.csv

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=9696

# Create non-root user for security
RUN adduser --disabled-password --gecos '' api-user && \
    chown -R api-user:api-user /app
USER api-user

# Expose the port the app runs on
EXPOSE 9696

# Command to run the application
CMD ["python", "predict.py"]