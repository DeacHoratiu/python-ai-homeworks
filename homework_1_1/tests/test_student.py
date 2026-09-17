import pytest
from pydantic import ValidationError
from homework_1_1.validator import StudentValidator

def test_valid_user():
    student_data = {
        "nume": "Ana Popescu",
        "varsta": 20,
        "email": "ana.popescu@example.com",
    }

    student = StudentValidator.model_validate(student_data)

    assert student.nume == "Ana Popescu"
    assert student.varsta == 20
    assert student.email == "ana.popescu@example.com"


def test_invalid_user():
    student_data = {
        "nume": "Ana Popescu",
        "varsta": 20,
        "email": "ana.popescu",
    }

    with pytest.raises(ValidationError):
        StudentValidator.model_validate(student_data)