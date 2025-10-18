from collections import defaultdict

data_dists = {
	'n0': {'0': 1, '1': 1, '2': 1},
	'n1': {'0': 1, '1': 1, '2': 1},
	'n2': {'0': 1, '1': 1, '2': 1},
	
	'n3': {'3': 1, '4': 1, '5': 1},
	'n4': {'3': 1, '4': 1, '5': 1},
	'n5': {'3': 1, '4': 1, '5': 1},
	
	'n6': {'6': 1, '7': 1, '8': 1, '9': 1},
	'n7': {'6': 1, '7': 1, '8': 1, '9': 1},
	'n8': {'6': 1, '7': 1, '8': 1, '9': 1},
	'n9': {'6': 1, '7': 1, '8': 1, '9': 1}
}

labels_totals = defaultdict(int)
for data_dist in data_dists.values():
	for label, quota in data_dist.items():
		labels_totals[label] += quota

results = defaultdict(list)
labels_pointers = defaultdict(int)
for node, data_dist in data_dists.items():
	for label, quota in data_dist.items():
		label_pointer = labels_pointers[label]
		label_total = labels_totals[label]
		
		results[node].append(f"{label}={label_pointer}-{label_pointer + quota}/{label_total}")
		
		labels_pointers[label] += quota
		assert labels_pointers[label] <= label_total

for node, result in results.items():
	print(f"{node}: fixed-{','.join(result)}")
