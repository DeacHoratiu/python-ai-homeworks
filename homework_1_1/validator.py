from pydantic import BaseModel, EmailStr, Field

class StudentValidator(BaseModel):
    nume: str = Field(min_length=2, max_length=50)
    varsta: int = Field(gt=0)
    email: EmailStr

    # def __str__(self):
    #     return(f"{self.nume} is {self.varsta} years old. Email: {self.email}")
