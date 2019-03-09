'''
Unzip submission and check structure
'''
from zipfile import ZipFile
from os import listdir, mkdir
from os.path import isfile, join

import sys

f = sys.argv[1]
unzipped_folder = "unzipped_submission"  # sys.argv[2]

## Check folder structure
try:
    mkdir(unzipped_folder)
except OSError:
    print '%s folder exists' % unzipped_folder

assert f.endswith('.zip')
f_zip = ZipFile(f)
f_basename = f_zip.filename.rstrip('.zip')
f_basename_split = f_basename.split("_")
assert len(f_basename_split) == 2, "Please check folder name."
assert f_basename_split[1] == "assign4", "Please check folder name."
netid = f_basename_split[0]

output_path = join(unzipped_folder, f_basename)
try:
    mkdir(output_path)
except OSError:
    print '%s folder exists' % output_path
f_zip.extractall(output_path)

print "Zip file extracted. Netid: %s." % netid

listdir_output_path = listdir(output_path)
assert "report.pdf" in listdir_output_path, "report.pdf not found."

## Check language
code_folder_list = map(lambda l: l + "_code", ["c", "cpp", "java", "python2", "python3"])
language = None
for code_folder in code_folder_list:
    if code_folder in listdir_output_path:
        language = code_folder.split("_")[0]
        break
assert language is not None, "Valid code folder not found."
print "Asserted coding language: %s." % language


code_path = join(output_path, language + "_code")
if "c" in language:
    assert "Makefile" in listdir(code_path), "Makefile not found."
elif "java" == language:
    assert "classification" in listdir(code_path), "classification/ not found."
elif "python" in language:
    assert "DecisionTree.py" in listdir(code_path), "DecisionTree.py not found."
    assert "RandomForest.py" in listdir(code_path), "RandomForest.py not found."
else:
    raise

print "Format check passed!"



