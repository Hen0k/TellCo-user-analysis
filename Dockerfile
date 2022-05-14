from python:3.10.2

EXPOSE 8501

Add . /tellco_analysis

WORKDIR /tellco_analysis

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt



CMD ["streamlit", "run", "app.py"]
