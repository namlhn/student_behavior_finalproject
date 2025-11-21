FROM python:3.11

# Attached information
LABEL author.name="HOAINAM" \
    author.email="namlh@dgk.vn"

#RUN apt-get install libcairo2-dev
ENV TZ="Asia/Bangkok"
RUN mkdir /www
WORKDIR /www
COPY requirements.txt /www/
#RUN pip install -U pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . /www
