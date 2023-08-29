# Use an official Python runtime as a parent image
FROM python:3.10.10

WORKDIR /app

# Set the working directory in the container to /app


# Copy the current directory contents into the container at /app


# Install any needed packages specified in requirements.txt

COPY requirements.txt ./
RUN pip install -r requirements.txt

RUN pip install gpt_index==0.4.15
RUN pip install --upgrade langchain
RUN pip install langchain==0.0.256
RUN pip install llama-index


COPY . ./

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["python", "app.py"]
