demo_survey = {
    "q1": {
        "type": "text-entry",
        "content": "What is your name?",
        "action": "text entry",
        "instruction": "Output your answers with reasons within 20 words."
    },
    "q2": {
        "type": "single-choice",
        "content": "What is your favorite fruit?",
        "action": "A. apple, B. banana, C. lemon",
        "instruction": "Output your single choice directly without reasons. Your output format should exactly like: choice: [choice]."
    },
    "q3": {
        "type": "multiple-choice",
        "content": "What do you eat everyday?",
        "action": "A. noodles, B. rice, C. meat",
        "instruction": "Output your multiple choices directly with reasons. Your output format should be exactly like: choice: [choice], reason: []. For example, if you want to select A and C, your output should be: choice: A,C, reason: [my reasons...]."
    }
}

demo_user_persona = {
    "u1": {
        "age": "24 years old",
        "job": "engineer"
    },
    "u2": {
        "education": "high school"
    },
    "u3": {
        "name": "Tom",
        "major": "Computer Science",
        "degree": "Master"
    }
}