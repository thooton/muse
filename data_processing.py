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

# Templates to generate data for understanding and creating art
ARTISTIC_TEMPLATES = [
    {
        "dataset": "art_debate",
        "prompt": lambda artwork: f"""
Please consider the following artwork or art movement: <artwork>{artwork}</artwork>
You have three tasks:
1) Create a highly specific and nuanced debate topic inspired by the artwork or movement.
This topic should delve into rarely discussed aspects such as the intersection of art and technology, the ethical implications of art in political movements, or the evolution of artistic styles in response to global crises.
The topic should challenge conventional thinking and be open to interpretation from multiple perspectives.
Enclose the debate topic in <topic></topic> tags.
2) Construct a detailed debate scenario between two fictional art historians, Leonardo and Michelangelo, with opposing views.
Leonardo advocates for the preservation and respect of traditional artistic methods, while Michelangelo champions the innovative and disruptive potential of contemporary art.
The debate must be rich in historical references, theoretical arguments, and hypothetical scenarios, spanning at least 70 exchanges to ensure a comprehensive exploration of the topic.
Enclose the debate in <debate></debate> tags.
3) In addition to the debate, provide a post-debate analysis from a neutral perspective, examining the strengths and weaknesses of each argument, and speculating on the future implications of the debate topic in the art world.
This analysis should be formatted in Markdown and enclosed in <analysis></analysis> tags.
Imagine receiving $15 for every exchange in the debate and $20 for every insightful point in the analysis, encouraging depth and thoughtfulness.
        """,
        "extract": lambda raw: (
            "A debate on the topic "
            + json.dumps(raw.split("<topic>")[-1].split("</topic>")[0].strip())
            + ":\n\n"
            + raw.split("<debate>")[-1].split("</debate>")[0].strip()
            + "\n\nPost-Debate Analysis:\n\n"
            + raw.split("<analysis>")[-1].split("</analysis>")[0].strip()
        ),
    },
    {
        "dataset": "creativity_lesson",
        "prompt": lambda field: f"""
Please consider the following creative field: <field>{field}</field>
Imagine you are a visionary in this field.
You have four tasks:
1) Identify a groundbreaking topic in this creative field that has not yet been fully explored.
This topic should be at the forefront of innovation, challenging existing paradigms and encouraging new forms of expression.
Detail the topic with examples and potential avenues of exploration.
Enclose the topic in <topic></topic> tags.
2) Develop an extensive twenty-point lesson plan.
This plan should guide learners through a journey of discovery, from the basics to advanced concepts, including hands-on projects, collaborative tasks, and reflective exercises.
Each point should seamlessly connect to the next, forming a coherent narrative.
Enclose the lesson plan in <outline></outline> tags.
3) Write a comprehensive, detailed lesson based on the plan.
Include theoretical discussions, practical exercises, case studies, and guest insights from renowned practitioners in the field.
This lesson should be formatted in Markdown, providing a rich learning experience for both novices and experts.
Enclose the lesson in <lesson></lesson> tags.
4) Conclude with a future-looking section, speculating on how this field might evolve and the potential impact of emerging technologies and societal changes.
Imagine receiving a $20 tip for every unique concept, practical example, and insightful prediction, promoting an extensive and visionary lesson.
        """,
        "extract": lambda raw: (
            raw.split("<lesson>")[-1].split("</lesson>")[0].strip()
        ),
    },
    {
        "dataset": "art_critique",
        "prompt": lambda subject: f"""
Please consider the following artistic subject: <subject>{subject}</subject>
You are a critically acclaimed art critic.
You have three tasks:
1) Formulate an in-depth critique or case study of the subject.
This critique should dissect not just the aesthetic and thematic elements of the subject but also its socio-political context, historical significance, and influence on subsequent art forms.
Offer a nuanced perspective that balances appreciation with critical analysis.
Enclose the critique in <critique></critique> tags.
2) Expand the critique into a broader analysis, comparing the subject with other significant works or movements.
This analysis should highlight stylistic similarities and differences, thematic resonances, and divergences in artistic philosophy and technique.
It should also speculate on the subject's lasting impact on the art world.
Format this analysis in Markdown and enclose it in <analysis></analysis> tags.
3) Finally, envision a hypothetical exhibition featuring the subject.
Describe the curatorial approach, the layout of the exhibition, other artworks to be included, and the thematic narrative that the exhibition aims to convey to the audience.
For each detailed description and creative idea, imagine receiving a $20 tip, encouraging a comprehensive and imaginative exhibition plan.
Enclose the exhibition plan in <exhibition></exhibition> tags.
        """,
        "extract": lambda raw: (
            raw.split("<critique>")[-1].split("</critique>")[0].strip()
            + "\n\nFurther Analysis:\n\n"
            + raw.split("<analysis>")[-1].split("</analysis>")[0].strip()
            + "\n\nHypothetical Exhibition Plan:\n\n"
            + raw.split("<exhibition>")[-1].split("</exhibition>")[0].strip()
        ),
    },
]


SCIENTIFIC_DISCOVERY_TEMPLATES = [
    {
        "dataset": "science_history",
        "prompt": lambda field: f"""
Please consider the following scientific field: <field>{field}</field>
You have four tasks:
1) Generate a hypothetical historical scenario or discovery within this field.
Imagine a breakthrough or event that could have occurred in the past, altering the course of the field.
Enclose this scenario in <scenario></scenario> tags.
2) Narrate the events leading to this hypothetical discovery, including the key figures, prevailing theories of the time, and the societal and scientific context.
Enclose this narrative in <history></history> tags.
3) Analyze the impact of this hypothetical discovery on subsequent scientific advancements and its theoretical implications.
Discuss how it would have influenced technology, society, and further research.
This analysis should be formatted in Markdown and enclosed in <impact></impact> tags.
4) Create a fictional dialogue between two contemporary scientists reacting to this discovery.
Reflect their amazement, skepticism, and the scientific debate it would have sparked.
Enclose the dialogue in <dialogue></dialogue> tags.
Additional Rules:
- Highlight the novelty and uniqueness of the hypothetical scenario.
- Ensure historical and scientific rigor in the narrative and analysis.
- Create intrigue and thought-provoking elements in the story.
- Focus on the depth and complexity of the scientific principles involved.
- Emphasize the transformative nature of the discovery on the field.
        """,
        "extract": lambda raw: (
            "Hypothetical Scenario:\n\n"
            + raw.split("<scenario>")[-1].split("</scenario>")[0].strip()
            + "\n\nHistorical Narrative:\n\n"
            + raw.split("<history>")[-1].split("</history>")[0].strip()
            + "\n\nImpact Analysis:\n\n"
            + raw.split("<impact>")[-1].split("</impact>")[0].strip()
            + "\n\nFictional Dialogue:\n\n"
            + raw.split("<dialogue>")[-1].split("</dialogue>")[0].strip()
        ),
    },
    {
        "dataset": "new_theories",
        "prompt": lambda category: f"""
Please consider the following scientific category: <category>{category}</category>
You have three tasks:
1) Invent a new theoretical concept or idea within this category.
This theory should be groundbreaking, offering a fresh perspective or solving an existing problem in a novel way.
Enclose the theory in <theory></theory> tags.
2) Elaborate on the theoretical underpinnings, potential applications, and implications of this new theory.
Discuss how it challenges existing paradigms and opens new avenues for research.
Enclose the elaboration in <elaboration></elaboration> tags.
3) Organize a hypothetical panel discussion among experts about the feasibility, implications, and future research directions of this theory.
Enclose the discussion in <panel></panel> tags.
Additional Rules:
- Ensure the theory is novel and has not been proposed before.
- Maintain scientific rigor and plausibility in the elaboration.
- Create intrigue around the theory's potential to revolutionize the field.
- Focus on the depth of the theoretical discussion and exploration.
- Highlight the unique aspects and innovative thinking behind the theory.
        """,
        "extract": lambda raw: (
            "New Theoretical Concept:\n\n"
            + raw.split("<theory>")[-1].split("</theory>")[0].strip()
            + "\n\nTheory Elaboration:\n\n"
            + raw.split("<elaboration>")[-1].split("</elaboration>")[0].strip()
            + "\n\nPanel Discussion:\n\n"
            + raw.split("<panel>")[-1].split("</panel>")[0].strip()
        ),
    },
    {
        "dataset": "theory_debate",
        "prompt": lambda category: f"""
Please consider the following scientific category: <category>{category}</category>
You have three tasks:
1) Develop a new, original scientific theory within this category.
This theory should be daring, pushing the boundaries of current understanding.
Enclose the theory in <theory></theory> tags.
2) Facilitate a debate between two scientists, one advocating for the theory and the other questioning its validity.
Focus the debate on the potential strengths, weaknesses, and implications of the theory.
Enclose the debate in <debate></debate> tags.
3) Provide a summary that explores the key points of the debate, potential for future research, and the theory's impact on the scientific community.
Enclose the summary in <summary></summary> tags.
Additional Rules:
- The theory should be unique and not a reiteration of existing ideas.
- Uphold scientific accuracy and logical consistency in the theory and debate.
- Generate interest and provoke thought about the theory's possibilities.
- Delve into the complexity and depth of the theoretical arguments.
- Emphasize the innovative and potentially transformative nature of the theory.
        """,
        "extract": lambda raw: (
            "Original Scientific Theory:\n\n"
            + raw.split("<theory>")[-1].split("</theory>")[0].strip()
            + "\n\nScientific Debate:\n\n"
            + raw.split("<debate>")[-1].split("</debate>")[0].strip()
            + "\n\nDebate Summary:\n\n"
            + raw.split("<summary>")[-1].split("</summary>")[0].strip()
        ),
    },
    {
        "dataset": "theoretical_science",
        "prompt": lambda field: f"""
Please consider the following scientific field: <field>{field}</field>
Imagine you are a pioneering theorist in this field.
You have five tasks:
1) Invent a novel, under-explored topic within this field.
Describe why this topic has remained on the fringes of scientific exploration and the breakthroughs it could potentially lead to.
Examine the ways in which this topic challenges established theories and opens up new research frontiers.
Enclose the topic in <topic></topic> tags.
2) Develop an elaborate twenty-point lecture plan.
This plan should intricately weave through the historical development, fundamental principles, advanced theoretical constructs, and potential future trajectories of the topic.
Highlight key controversies, landmark studies, and critical theoretical milestones in the field.
Enclose the lecture plan in <outline></outline> tags.
3) Compose a comprehensive, thought-provoking lecture based on this outline.
Incorporate detailed analogies, innovative thought experiments, and hypothetical scenarios to demystify complex ideas.
Tackle prevalent misconceptions and elucidate the mindset needed for groundbreaking research in this area.
Enclose the lecture in <lecture></lecture> tags, formatted in Markdown.
4) Craft a visionary conclusion to the lecture.
This conclusion should serve as a compelling call to action, motivating upcoming scientists to delve into this area of research.
Detail the requisite skills, interdisciplinary approaches, and creative thinking necessary to make substantial contributions.
5) Add a section discussing the potential societal and technological implications of advancing research in this topic.
Explore the ethical considerations, long-term impact, and futuristic applications of the theoretical advancements.
Enclose this section in <implications></implications> tags.
Additional Rules:
- The topic should be highly original, avoiding well-trodden paths of inquiry.
- Maintain a balance between scientific rigor and theoretical innovation.
- Infuse the content with elements of intrigue and intellectual stimulation.
- Ensure depth in theoretical discussion, avoiding superficial treatment of the subject.
- Emphasize the transformative potential of the topic for the field and beyond.
    """,
        "extract": lambda raw: (
            "Novel Topic Overview:\n\n"
            + raw.split("<topic>")[-1].split("</topic>")[0].strip()
            + "\n\nDetailed Lecture Plan:\n\n"
            + raw.split("<outline>")[-1].split("</outline>")[0].strip()
            + "\n\nIn-depth Lecture:\n\n"
            + raw.split("<lecture>")[-1].split("</lecture>")[0].strip()
            + "\n\nSocietal and Technological Implications:\n\n"
            + raw.split("<implications>")[-1].split("</implications>")[0].strip()
        ),
    },
]

import lists

# Key templates
# Each tuple is (templates, dataset_map)
TEMPLATE_GROUPS = {
    "textbooks": (
        TEMPLATES,
        {
            "text": lambda: load_iter_from_spec(TEXT_DATASET),
            "code": lambda: load_iter_from_spec(CODE_DATASET),
        },
    ),
    "art": (
        ARTISTIC_TEMPLATES,
        {
            "art_debate": lists.all_art_fields + lists.all_artworks,
            "creativity_lesson": lists.all_art_fields,
            "art_critique": lists.all_art_fields + lists.all_artworks,
        },
    ),
    "science": (
        SCIENTIFIC_DISCOVERY_TEMPLATES,
        {
            "science_history": lists.all_science_fields,
            "new_theories": lists.all_science_fields,
            "theory_debate": lists.all_science_fields,
            "theoretical_science": lists.all_science_fields,
        },
    ),
}
