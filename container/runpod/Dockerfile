FROM shinoharahare/stylegan_xl AS stage-0

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt update

RUN apt install -y --no-install-recommends \ 
    wget \
    curl \
    p7zip-full \
    openssh-server \
    git

RUN pip install ipykernel

# Google Cloud CLI
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt update && \
    apt install -y --no-install-recommends google-cloud-cli

# Code Server
RUN curl -fsSL https://code-server.dev/install.sh | sh

RUN apt clean && rm -rf /var/lib/apt/lists/*

FROM stage-0

WORKDIR /workspace

ADD start.sh /scripts/
ADD start.py /scripts/
RUN chmod +x /scripts/start.sh
CMD ["/scripts/start.sh"]

EXPOSE 80
