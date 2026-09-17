import json
from pathlib import Path

from homework_1_1.validator import StudentValidator

current_dir = Path(__file__).parent
student_json = current_dir / "data" / "student.json"

with open(student_json, "r") as json_file:
    student_data = json.load(json_file)

validate_user = StudentValidator.model_validate(student_data)

print(validate_user.model_dump_json(indent=2))