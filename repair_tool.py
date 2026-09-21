import json_repair

json_string = "[1,2,3,"

# repairs the JSON and returns the parsed Python object (a list in this case)
repaired_obj = json_repair.loads(json_string)
print(repaired_obj)
# Output: [1, 2, 3]

# Or, if you just want the repaired JSON string back:
repaired_string = json_repair.repair_json(json_string)
print(repaired_string)
# Output: [1,2,3]
