import numpy as np
import csv


#Statistics und so
data = None

with open('transformations.csv', newline='') as f:
    reader = csv.reader(f, delimiter='\t')
    data = list(reader)

# sift_normal = []
# orb_normal = []
# sift_filter = []
# orb_filter = []

for i in data:
    if len(i[3]) < 3:
        i[3] = "0"
    if len(i[5]) < 3:
        i[5] = "0"
#     if "normal" in i[0] and "normal" not in i[1]:
#         sift_normal.append(float(i[3].replace(',','.')))
#         orb_normal.append(float(i[5].replace(',','.')))
#     if "Filter" in i[0] and "normal" in i[1]:
#         sift_filter.append(float(i[3].replace(',','.')))
#         orb_filter.append(float(i[5].replace(',','.')))

# print(np.median(sift_normal))
# print(np.median(orb_normal))
# print(np.median(sift_filter))
# print(np.median(orb_filter))
#10.3783523688349
#12.7140575457968
#9.24944347232242
#13.6863899923812

