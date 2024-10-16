from gettext import find
from os import listdir


def find_csv_filenames(path_to_dir, suffix=".csv", surfix='predicted'):
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix) and filename.startswith(surfix)]


filenames = find_csv_filenames('./')
base_result = filenames.index("predicted-count-cwb-ng45.csv")
print(filenames)
del filenames[base_result]
print(base_result)
print(filenames)
# for index, name in filenames:
#     print(index, name)
# with open('old.csv', 'r') as t1, open('new.csv', 'r') as t2:
#     fileone = t1.readlines()
#     filetwo = t2.readlines()

# with open('update.csv', 'w') as outFile:
#     for line in filetwo:
#         if line not in fileone:
#             outFile.write(line)
