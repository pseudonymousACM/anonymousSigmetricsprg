#!/usr/bin/env sh

set -e

DNSMASQ_CACHE_SIZE="${DNSMASQ_CACHE_SIZE:-1000}"
sed "s/\$DNSMASQ_CACHE_SIZE/${DNSMASQ_CACHE_SIZE}/g" /etc/dnsmasq.conf.tmpl > /etc/dnsmasq.conf

exec dnsmasq --no-daemon