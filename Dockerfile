#DOCKER FILE
FROM python:3.10
# Allow statements and log messoges to inmediat
ENV PYTHONUNBUFFERED True

RUN apt-get update && apt-get install -y libgl1-mesa-glx

ENV PORT=8081
# Copy local code to the container image.
ENV APP_HOME /back-end
WORKDIR $APP_HOME
COPY . ./

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt
ENV NLTK_DATA /app/nltk_data

RUN python -m nltk.downloader -d $NLTK_DATA punkt

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 moodbot:app