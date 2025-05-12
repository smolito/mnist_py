FROM tensorflow/tensorflow:2.12.0-gpu

WORKDIR /mnist_py

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# RUN mkdir -p /mnist_py/output

COPY . .

CMD ["python", "src/mnist_conv.py"]