FROM python:3.6
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT [ "python" ]
CMD [ "main.py" ]