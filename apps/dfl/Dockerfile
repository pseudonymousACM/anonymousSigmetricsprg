FROM python:3.12.10-slim-bookworm

RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt/,sharing=locked \
  	--mount=type=cache,target=/var/lib/apt/,sharing=locked \
  		apt update && apt install -y git iproute2 dnsutils iptables iputils-ping gettext

WORKDIR /opt/p2p-search/dfl/

COPY ./requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip/,sharing=locked \
    pip install -r ./requirements.txt

COPY ./ ./

ENTRYPOINT ["./docker-entrypoint.sh"]