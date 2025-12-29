FROM python:3.8

WORKDIR /project

COPY requirements.txt ./

RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

RUN pip uninstall -y opencv-python && \
    pip install --no-cache-dir opencv-python-headless && \
    pip install --no-cache-dir --upgrade pip

COPY . ./

CMD ["python", "main.py"]
