# Pull any base image that includes python3
FROM python:3.10.8

# install the toolbox runner tools
RUN pip install json2args==0.4.0


# Do anything you need to install tool dependencies here
RUN pip install scikit-learn==1.1.3

# create the tool input structure
RUN mkdir /in
COPY ./in /in
RUN mkdir /out
RUN mkdir /src
COPY ./src /src

WORKDIR /src
CMD ["python", "run.py"]
