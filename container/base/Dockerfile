FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel AS stage-0

COPY requirements.txt .
RUN pip install --no-cache-dir --no-deps -r requirements.txt
RUN rm requirements.txt

RUN pip install --no-cache-dir tensorboard protobuf==3.20.*

FROM stage-0
