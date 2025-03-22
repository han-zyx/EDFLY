FROM alpine:latest

WORKDIR /data

COPY ./ /data/

CMD ["ls", "/data"]
