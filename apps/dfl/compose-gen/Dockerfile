FROM python:3.12.10-slim-bookworm

WORKDIR /opt/p2psims/dfl/compose-gen/

COPY ./requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip/,sharing=locked \
    pip install -r ./requirements.txt

COPY ./ ./

ENTRYPOINT ["./main.py"]