# In path/to/your/dev/folder/Dockerfile
# Base Image
FROM python:3.6

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev \
    pip install -U libra

EXPOSE 8000

CMD python -c "print('works')" 