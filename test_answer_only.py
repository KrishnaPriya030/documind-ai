
from app.answer_generator import generate_answer

question = (
    "According to the University Fact Sheet, how many "
    "schools and departments does the University of Kerala have?"
)

retrieved_chunks = [
    {
        "doc": "academic_regulations.pdf",
        "page": 14,
        "text": (
            "Teaching and Research: The University of Kerala has "
            "its own departments of teaching and research. The "
            "forty-three departments of teaching and research "
            "are organized as nine schools."
        ),
    },
    {
        "doc": "academic_regulations.pdf",
        "page": 21,
        "text": (
            "University Fact Sheet: No. of Departments 43 "
            "(Science and Technology: 18; Social Sciences: 11; "
            "Arts and Humanities: 14)."
        ),
    },
]

print("Question:", question)
print("\nTesting answer generation with relevant context...\n")

answer = generate_answer(question, retrieved_chunks)

print("\nAnswer:")
print(answer)
