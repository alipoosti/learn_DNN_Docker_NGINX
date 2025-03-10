# Import nginx image
FROM nginx:latest

# Copy our own config files, and replaces them with default ones
RUN rm /etc/nginx/nginx.conf
COPY nginx.conf /etc/nginx/
RUN rm /etc/nginx/conf.d/default.conf
COPY project.conf /etc/nginx/conf.d/