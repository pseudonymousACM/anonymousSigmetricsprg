#!/usr/bin/env sh

set -e

until getent hosts dnsmasq; do
	echo "Waiting for dnsmasq..."
	sleep 1 # TODO: this should be a random exponential wait, up to e.g. 7 seconds.
done

DNSMASQ_IP="$(getent hosts dnsmasq | awk '{ print $1 }')" # TODO: this does produce a redundant output to the console.
export DNSMASQ_IP
envsubst < ./dns/resolv.conf.tmpl > /etc/resolv.conf

python "$@"