FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the model at build time so the image is ready to serve immediately.
# (In a bigger real-world setup, training and serving would be separate
# pipelines/images — see README "What I'd Add Next".)
RUN python train.py

EXPOSE 5001

CMD ["python", "app.py"]
