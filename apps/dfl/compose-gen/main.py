#!/usr/bin/env python3

from os import environ

import ruamel.yaml

NODES_COUNT = int(environ.get('NODES_COUNT', '10'))  # Borrowed from `conf.py`; the default value should match with there.
FAIR_SPREAD_ON_GPUS = environ.get('FAIR_SPREAD_ON_GPUS', None)
if FAIR_SPREAD_ON_GPUS is not None:
	FAIR_SPREAD_ON_GPUS = FAIR_SPREAD_ON_GPUS.lower()
	assert FAIR_SPREAD_ON_GPUS in ('id', 'visible')

DEVICE = environ.get('DEVICE', None)  # Borrowed from `conf.py`.
PREPARE_DATA_DISTRIBUTION = len(environ.get('PREPARE_DATA_DISTRIBUTION', '')) > 0

BUILD_DIR = environ.get('BUILD_DIR', '.')
LOGS_DIR = environ.get('LOGS_DIR', '.')
DATA_DIR = environ.get('DATA_DIR', './data')
REPO_DIR = environ.get('REPO_DIR', '.')
RUN_DIR = environ.get('RUN_DIR', '.')

gpus_count = None
if FAIR_SPREAD_ON_GPUS is not None:
	if DEVICE is not None:
		raise ValueError(f"Fair spread on GPUs whilst DEVICE={DEVICE} is set is prohibited.")
	
	import gpustat
	
	gpus_count = gpustat.gpu_count()

compose = {
	'services': {
		'node-image-build': {
			'build': BUILD_DIR,
			'image': 'localhost/p2p-search/dfl/node',
			'pull_policy': 'build',
			'restart': 'no',
			'entrypoint': ['sh', '-c'],
			'command': ['exit', '0']
		},
		'data': {
			'image': 'localhost/p2p-search/dfl/node',
			'pull_policy': 'never',
			'command': './data.py',
			'environment': {
				'DATA_DOWNLOAD': True,
				'INDEX': '0',
				'PREPARE_DATA_DISTRIBUTION': True if PREPARE_DATA_DISTRIBUTION else ''
			},
			'env_file': ['./.env'],
			'volumes': [
				f'{DATA_DIR}:/opt/p2p-search/dfl/data/',
				f'{LOGS_DIR}:/opt/p2p-search/dfl/logs/',
				f'{RUN_DIR}/data_dist/:/opt/p2p-search/dfl/run/data_dist/',
			],
			'depends_on': {
				'node-image-build': {
					'condition': 'service_started'
				},
				'dnsmasq': {
					'condition': 'service_started'  # TODO: upgrade to health-check.
				}
			}
		},
		'otel-collector': {
			'image': 'ghcr.io/open-telemetry/opentelemetry-collector-releases/opentelemetry-collector-contrib:0.125.0',
			'restart': 'on-failure',
			'user': '0:0',
			'volumes': [
				f'{REPO_DIR}/otel-collector-conf.yaml:/etc/otelcol-contrib/config.yaml',  # TODO: embed this inside the compose file or the image.
				f'{LOGS_DIR}:/var/log/otel-collector/'
			]
		},
		'dnsmasq': {
			'image': 'dockurr/dnsmasq:2.91',
			'restart': 'on-failure',  # TODO: add health-check.
			'volumes': [  # TODO: embed this inside the compose file or the image.
				f'{REPO_DIR}/dns/dnsmasq.conf.tmpl:/etc/dnsmasq.conf.tmpl',
				f'{REPO_DIR}/dns/dnsmasq-docker-entrypoint.sh:/opt/dnsmasq/docker-entrypoint.sh'
			],
			'entrypoint': ['/sbin/tini', '--', f'/opt/dnsmasq/docker-entrypoint.sh'],
			'env_file': ['./.env']
		}
	},
}

nx = ruamel.yaml.CommentedMap({
	'image': 'localhost/p2p-search/dfl/node',
	'command': './main.py',
	'pull_policy': 'never',
	'restart': 'no',
	'env_file': ['./.env'],
	'depends_on': {
		'data': {
			'condition': 'service_completed_successfully'
		},
		'node-image-build': {
			'condition': 'service_started'
		},
		'dnsmasq': {
			'condition': 'service_started'  # TODO: upgrade to health-check.
		}
	},
	'volumes_from': ['data'],
	'deploy': {
		'resources': {
			'reservations': {
				'devices': [
					{
						'driver': 'nvidia',
						'count': 'all',
						'capabilities': ['gpu']
					}
				]
			}
		}
	},
	'cap_add': ['NET_ADMIN']
})

nx_env = ruamel.yaml.CommentedMap({
	'DATA_DOWNLOAD': False
})
if PREPARE_DATA_DISTRIBUTION:
	nx_env['DATA_DISTRIBUTION'] = 'prepared-file'
if FAIR_SPREAD_ON_GPUS is not None and FAIR_SPREAD_ON_GPUS == 'visible':
	nx_env['DEVICE'] = 'cuda'

nx_env.yaml_set_anchor('nx_env')  # FIXME: the output uses the weird syntax of `<<: &x-nx`, that works but it is unexpected.
nx['environment'] = nx_env

compose['x-nx'] = nx
nx.yaml_set_anchor('x-nx')

for i in range(0, NODES_COUNT):
	name = f'n{i}'
	
	ni = ruamel.yaml.CommentedMap({
		'hostname': name
	})
	
	ni_env = ruamel.yaml.CommentedMap({
		# NOP.
	})
	if FAIR_SPREAD_ON_GPUS is not None:
		if FAIR_SPREAD_ON_GPUS == 'id':
			ni_env['DEVICE'] = f'cuda:{i % gpus_count}'
		elif FAIR_SPREAD_ON_GPUS == 'visible':
			ni_env['CUDA_VISIBLE_DEVICES'] = str(i % gpus_count)
		else:
			assert False
	
	ni_env.add_yaml_merge([(0, nx_env)])
	ni['environment'] = ni_env
	
	ni.add_yaml_merge([(0, nx)])
	compose['services'][name] = ni

compose['services']['n0']['networks'] = {
	'default': {
		'aliases': ['server', 'cdfl']  # For CFL and CDFL.
	}
}

with open('./compose.yaml', 'w') as f:
	ruamel.yaml.YAML().dump(compose, f)
