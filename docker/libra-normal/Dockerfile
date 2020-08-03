# In path/to/your/dev/folder/Dockerfile
# Base Image
FROM python:3.6

# COPY requirements.txt docker/requirements.txt

# RUN pip freeze > requirements.txt/ pip install -r requirements.txt/

RUN pip install -U libra
# ENV PYTHONPATH='/src/:$PYTHONPATH'

# WORKDIR /data

EXPOSE 8000

# CMD python ./libra/queries.py


CMD python -c "print('docker image has been build')"
