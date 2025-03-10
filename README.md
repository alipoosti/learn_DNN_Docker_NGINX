**Author:** Alireza Poostindouz

**Contact:** alipoosti01@gmail.com

**Date:** Sept 8, 2023

# MODEL DEPLOYMENT DEMONSTRATION

In this part we have: 

- Explanation of how to create a container to process inference requests from a pretrained model in the huggingface model hub: [https://huggingface.co/models](https://huggingface.co/models). 
- Explanation of how to include server components to support multiple parallel incoming requests. 
- A demonstration of how POST requests can be made to the container endpoint and print out the response

Files and source codes for the above parts are each separated in the three folders: 
`my_inference_app` , `nginx`, and `post_request`. 

## The inference app

First, let's choose a suitable model for demonstrating a common ML inference task. To do so, I chose *Image Classification* task and chose  `ResNet50` from [https://huggingface.co/microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50) from model hub since it is one of the most popular models for image classification. Another reason as to why I chose this task is I want to test ResNet on multiple pictures I have from my own lovely cat **Cammie**. Here is an example photo of her. 😍

![My cat, Cammie](./post_request/cammie.jpg)

To implement the inference, I chose to use 
`PyTorch`  and `flask` for converting the ML model inference code into a web app.  

First, we write the ML inference python code as in `my_inference_app/ResNet50_inference.py`. We make sure to define the `inference` function so we can import it to the next part. We can use the code provided in ResNet's model hub [page](https://huggingface.co/microsoft/resnet-50) as our base for the source code but the image processor used there is deprecated so we use the standard processor that was provide by PyTorch in [PyTorch Hub](https://pytorch.org/hub/pytorch_vision_resnet/). 

```Python
import sys, json
from transformers import ResNetForImageClassification
import torch
from torchvision import transforms
from PIL import Image

def inference(image):
    """Run inference using ResNet50 on an input PIL image

    Parameters:
    image : input PIL image

    Returns:
    prediction result in json dictionary format
    """

    # loading the pretrained model
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

    # Processing the PIL image to correct format to input to model

    # All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # get model output in form of logits
    with torch.no_grad():
        logits = model(input_batch).logits

    # find the associated label for prediction
    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()

    # response
    response = model.config.id2label[predicted_label]
    return {
        "statusCode": 200,
        "body": json.dumps(response)
    }

if __name__ == '__main__':
    img_path = sys.argv[1]
    image = Image.open(img_path)
    result = inference(image)
    print(result)
```

To test that ML inference is working we create a virtual environment, install required packages (see `my_inference_app/requirements.txt`) and run the following command:

```bash
python3 my_inference_app/ResNet50_inference.py post_request/cammie.jpg
```

we should get the following response:

```
{'statusCode': 200, 'body': '"tabby, tabby cat"'}
```

that is in an appropriate format for an HTML response to a POST request. 

Now that the inference is working, let's make wrap our code into a web app format. We use flask library and for this we write the code as in `my_inference_app/server.py`. Make sure to add two routs for the web app, the home route to test if the app is running on the server and the main route for our purpose to POST inference request to the app. 

```Python
import flask
from PIL import Image
import io
from ResNet50_inference import inference
from flask import Flask
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Hello! Use /predict route to POST your prediction requests."

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    if flask.request.files.get("image"):
        image = flask.request.files["image"].read()
        image = Image.open(io.BytesIO(image))

        result = inference(image)

        data["response"] = result
        data["success"] = True
    return flask.jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
```

We will also add a WSGI gateway to give access to the inference web service (which is the proper thing to do.) 

```Python
from server import app

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
```

We can test if server is working after running

```bash
python3 my_inference_app/wsgi.py
```

We should get a message that the web app is up and running. Then we can use our browser and go to `http://localhost:80`. If the *Hello!* message is displayed then it means the app is working and ready to get inference POST requests.

The last step for this part is to create a Docker image for our inference app. We do so by creating the Dockerfile as in `my_inference_app/Dockerfile`. Note that using python as the base image, would not seamlessly allow installing PyTorch related libraries in the container. We can use the NVIDIA's latest pytorch docker image as base. This base image's documentation can be found [here](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-08.html). This base image; however, is very large (more than 20GB), so I used the official PyTorch base from [docker hub](https://hub.docker.com/r/pytorch/pytorch), which is lighter than the NVIDIA's base image. This base image made installing and using the PyTorch libraries easier through the process of building the docker image. 

We can build this docker image without WSGI and NGINX components, by exposing the 8000 port and adding the command line at the end of the Dockerfile, but a proper way of building a web server that provides a service of a web app is through the application of NGINX and WSGI gateways. So make sure to comment lines 16 and 22 of `my_inference_app/Dockerfile`. 

```docker
# Use the official Python image as the base image
# FROM python:3.9
# FROM nvcr.io/nvidia/pytorch:23.08-py3
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /my_inference_app

# Copy requirements into the container
COPY  requirements.txt .

# Install requirements
RUN pip install -U pip
RUN pip install --no-cache-dir --default-timeout=900 -r requirements.txt

# Expose the port the app runs on
# EXPOSE 8000

# Copy all required files into the container
COPY . .

# Define the command to run the Python script
# CMD ["python3", "wsgi.py"]
```

## NGINX components

We need to configure the NGINX related settings of our web service. We use Gunicorn WSGI as a gateway to our backend inference engine and NGINX to handle parallel incoming requests. Gunicorn WSGI can be configured to utilize multiple workers of the web app in the backend but here I set the number of workers to only 1.  For each worker though we should set an appropriately large number of allowed connections. As a default I set 1024 concurrent connections but this can be changes through `nginx/nginx.conf`. 

```nginx
# Define the user that will own and run the Nginx server
user  nginx;
# Define the number of worker processes; recommended value is the number of
# cores that are being used by your server
worker_processes  1;
# Define the location on the file system of the error log, plus the minimum
# severity to log messages for
error_log  /var/log/nginx/error.log warn;
# Define the file that will store the process ID of the main NGINX process
pid        /var/run/nginx.pid;

# events block defines the parameters that affect connection processing.
events {
    # Define the maximum number of simultaneous connections that can be opened by a worker proce$
    worker_connections  1024;
}

# http block defines the parameters for how NGINX should handle HTTP web traffic
http {
    # Include the file defining the list of file types that are supported by NGINX
    include       /etc/nginx/mime.types;
    # Define the default file type that is returned to the user
    default_type  text/html;
    # Define the format of log messages.
    log_format  main  '$remote_addr - $remote_user [$time_local] "$request" '
                      '$status $body_bytes_sent "$http_referer" '
                      '"$http_user_agent" "$http_x_forwarded_for"';
                          # Define the location of the log of access attempts to NGINX
    access_log  /var/log/nginx/access.log  main;
    # Define the parameters to optimize the delivery of static content
    sendfile        on;
    tcp_nopush     on;
    tcp_nodelay    on;
    # Define the timeout value for keep-alive connections with the client
    keepalive_timeout  65;
    # Define the usage of the gzip compression algorithm to reduce the amount of data to transmit
    #gzip  on;
    # Include additional parameters for virtual host(s)/server(s)
    include /etc/nginx/conf.d/*.conf;
}
```

Make sure to also add `nginx/project.conf` and  `nginx/Dockerfile` in the NGINX folder. The Dockerfile image role for this part is to make sure we replace our desired configurations with the default configurations of the NGINX base image. 

```nginx
server {

    listen 80;
    server_name docker_flask_gunicorn_nginx;

    location / {
        proxy_pass http://my_inference_app:8000;

        # Do not change this
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /static {
        rewrite ^/static(.*) /$1 break;
        root /static;
    }
}
```

## Put everything together

The last step for this part is to combine the two services with docker. I chose port 8000 for accessing the inference web app, and port 80 for accessing the NGINX web proxy. By POSTing request to the NGINX web server through port 80, NGINX forwards the requests through port 8000 to the Gunicorn WSGI gateway of the inference app. Thus, we write the `docker-compose.yml` file in the home directory and run `run_docker.sh` bash script that kills other docker process and then starts building our two docker containers. 

```yml
version: '3'

services:
  my_inference_app:
    container_name: my_inference_app_ap
    restart: always
    build: ./my_inference_app
    image: alipoosti/resnet_inference:my_inference_app_ap
    ports:
      - "8000:8000"
    command: gunicorn -w 1 -b 0.0.0.0:8000 wsgi:app --timeout 2000
  
  nginx:
    container_name: nginx_ap
    restart: always
    build: ./nginx
    image: alipoosti/resnet_inference:nginx_ap
    ports:
      - "80:80"
    depends_on:
      - my_inference_app
```

```bash
#! /bin/bash

echo killing old docker processes
docker-compose rm -fs

echo building docker containers
docker-compose up --build -d
```

Note that structure of the files are of the form:

```
.
├── my_inference_app 
│   ├── requirements.txt  
│   ├── ResNet50_inference.py       
│   ├── wsgi.py
│   ├── server.py
│   └── Dockerfile
├── nginx
│   ├── nginx.conf          
│   ├── project.conf
│   └── Dockerfile
├── docker-compose.yml
└── run_docker.sh
```

## POST an inference request

When the container is running and ready to receive POST requests (check it by going to `http://localhost:80`) we can send a POST request in different ways. We can use `curl` like

```bash
curl -X POST -F "image=@/path/to/your/image.jpg" http://localhost:80/predict
```
or we can use the Python code as in `post_request/post_request.py`. Make sure to have the `request` package installed or create a virtual env to install it or simply run `post_request/pre_post_request.sh` before running:

```bash
python3 post_request/post_request.py ./path/to/image/file.jpg
```

For example by running this web service for the pictures we have in the `post_request` folder we get:


```
$ python3 post_request/post_request.py  ./post_request/cat.jpg

Checking results for ./post_request/cat.jpg
{'response': {'body': '"tabby, tabby cat"', 'statusCode': 200}, 'success': True}

$ python3 post_request/post_request.py  ./post_request/maltese.jpg

Checking results for ./post_request/maltese.jpg
{'response': {'body': '"Maltese dog, Maltese terrier, Maltese"', 'statusCode': 200}, 'success': True}

$ python3 post_request/post_request.py  ./post_request/cammie.jpg

Checking results for ./post_request/cat.jpg
{'response': {'body': '"tabby, tabby cat"', 'statusCode': 200}, 'success': True}
```

## Running containers using images from Docker Hub

To pull the docker images from docker hub run:

```bash
docker pull alipoosti/resnet_inference:nginx_ap
docker pull alipoosti/resnet_inference:my_inference_app_ap
docker-compose rm -fs
docker-compose up -d
```

then you can POST your inference requests to port 80 of localhost, for example:

```bash
curl -X POST -F "image=@/path/to/your/image.jpg" http://localhost:80/predict
```

or by using the `post_request/post_request.py` python script.

## References

1. https://pytorch.org/hub/pytorch_vision_resnet/
2. https://huggingface.co/microsoft/resnet-50
3. https://www.paepper.com/blog/posts/pytorch-gpu-inference-with-docker/
4. https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-08.html
5. https://hub.docker.com/r/pytorch/pytorch 
6. https://towardsdatascience.com/how-to-deploy-ml-models-using-flask-gunicorn-nginx-docker-9b32055b3d0 

