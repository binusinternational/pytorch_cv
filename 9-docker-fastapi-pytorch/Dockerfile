FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN apt-get update && apt-get -y install libgl1-mesa-glx

COPY ./requirements.txt $pwd

RUN pip install -r requirements.txt

COPY ./api /app/app

CMD ["/start-reload.sh"]