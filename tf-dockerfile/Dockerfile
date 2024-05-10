FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && \
    apt-get install sudo

RUN useradd --create-home appuser && \
    passwd -d appuser && \
    usermod -a -G sudo appuser

RUN pip install --upgrade pip --no-cache-dir && \
    pip install pandas && \
    pip install matplotlib && \
    pip install scipy && \
    pip install scikit-learn && \
    pip install tensorflow-datasets && \
    pip install pydot && \
    apt-get install graphviz



# RUN mkdir /code
WORKDIR /home/appuser/
COPY entrypoint.sh ./run.sh
RUN chmod 774 ./run.sh
USER appuser

ENV TF_ENABLE_ONEDNN_OPTS=0

ENTRYPOINT ["./run.sh"]

