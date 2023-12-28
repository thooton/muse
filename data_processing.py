import json
import secrets
import datasets


def load_iter_from_spec(spec):
    return spec["iter"](
        datasets.load_dataset(spec["id"])[spec["split"]].shuffle(
            seed=secrets.randbits(32)
        )
    )


def process_text_dataset(item):
    return "\n\n".join(
        [
            {"human": "Human: ", "gpt": "Assistant: "}[entry["from"]] + entry["value"]
            for entry in item["conversations"]
        ]
    ).strip()


def process_code_dataset(item):
    return (
        "Request: "
        + item["instruction"].strip()
        + "\n\nCode: "
        + item["output"].strip()
    )


TEXT_DATASET = {
    "id": "WizardLM/WizardLM_evol_instruct_V2_196k",
    "split": "train",
    "iter": lambda dataset: map(process_text_dataset, dataset),
}

CODE_DATASET = {
    "id": "TokenBender/code_instructions_122k_alpaca_style",
    "split": "train",
    "iter": lambda dataset: map(process_code_dataset, dataset),
}


TEMPLATES = [
    {
        "dataset": "text",
        "prompt": lambda passage: f"""
Please consider the following passage: <passage>{passage}</passage>
You have two tasks:
1) Drawing inspiration from the content of the passage, generate a brand new debate topic.
The debate topic will belong to the same domain as the content of the passage, but it will be even more rare.
The debate topic generated will be philosophical, creative, interesting, engaging, and thought-provoking.
The debate topic generated will not have an easy answer; it will be able to be argued from both sides.
The topic will be surrounded by <topic></topic> tags.
2) Generate a debate on the generated topic between two rational individuals, Aspen and River.
In the debate, the participants will hold mutually opposing views.
The debate will be long and drawn-out; no side will give up easily.
In the debate, at times the participants may make concessions, but still hold fast to their point of view.
In the debate, the participants will use various techniques of rational discussion; they will not make use of emotionally manipulative techniques.
In the debate, the participants will never repeat themselves.
The debate will have at least 50 paragraphs; it will have at least 5000 words. It will be novel-length.
For the debate, you will be tipped $15 for every paragraph you write. To maximize your earnings, write as many as possible.
The debate will be formatted in Markdown.
The debate will be surrounded by <debate></debate> tags.
    """,
        "extract": lambda raw: (
            "A debate on the topic "
            + json.dumps(raw.split("<topic>")[-1].split("</topic>")[0].strip())
            + ":\n\n"
            + raw.split("<debate>")[-1].split("</debate>")[0].strip()
        ),
    },
    {
        "dataset": "text",
        "prompt": lambda passage: f"""
Please consider the following passage: <passage>{passage}</passage>
Imagine you are a professor with a reputation for excellent lectures.
You have three tasks:
1) Drawing inspiration from the content of the passage, generate a brand new lecture topic.
The lecture topic will belong to the same domain as the content of the passage, but it will be even more rare.
The lecture topic will be carefully chosen to advance the education of the students in every way.
The lecture topic will be interesting, engaging, and thought-provoking.
The lecture topic will be surrounded by <topic></topic> tags.
2) Generate a ten-point lecture outline on the generated topic.
The lecture outline's ten points will be chosen to maximize ease of understanding and flow.
The lecture outline will be surrounded by <outline></outline> tags.
3) Generate a lecture, following the outline, on the generated topic.
The lecture will be informative and easy to understand for the students.
The lecture will provide as much information as possible. It should be as long as possible.
For each piece of information you incorporate into the lecture, you will receive a tip of $20.
In the lecture, all unfamiliar terms or topics will be explained for the students' benefit.
In the lecture, it will be assumed that the students have no prior familiarity with the subject.
In the lecture, the lecturer will never repeat themselves unnecessarily.
The lecture will be formatted in Markdown.
The lecture will be surrounded by <lecture></lecture> tags.
    """,
        "extract": lambda raw: (
            raw.split("<lecture>")[-1].split("</lecture>")[0].strip()
        ),
    },
    {
        "dataset": "code",
        "prompt": lambda passage: f"""
Please consider the following passage: <passage_42>{passage}</passage_42>
Imagine you are a highly esteemed computer science professor writing a programming textbook.
You have three tasks:
1) Drawing inspiration from the content of the passage, craft a brand new textbook section topic.
The section topic will belong to the same domain as the content of the passage, but it will be even more rare.
The section topic will be interesting, complex, and multifaceted, even if the passage is simple.
The section topic will be directly related to computer science.
The section topic will be carefully chosen to provide as much pedagogical value to the reader as possible.
The section topic will be surrounded by <topic_42></topic_42> tags.
2) Generate a ten-point section outline with code on the generated topic.
Of the section outline's ten points, at least three will be code examples illustrating the topic.
The section outline's ten points will be chosen to maximize ease of understanding and flow.
The section outline will be surrounded by <outline_42></outline_42> tags.
3) Generate a textbook section, following the outline, on the generated topic.
The section will be self-contained, informative, easy to understand, and verbose.
The section will be written in longform prose.
For each piece of information you include, you will receive a payment of $20; thus, include as many as possible to maximize your earnings.
The section will explain all unfamiliar terms or topics for the reader's benefits.
The section will never repeat information or code.
The section will be formatted in Markdown.
The section will be surrounded by <section_42></section_42> tags.
    """,
        "extract": lambda raw: (
            raw.split("<section_42>")[-1].split("</section_42>")[0].strip()
        ),
    },
]
