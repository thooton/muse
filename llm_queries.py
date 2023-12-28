from config import API_KEY, API_ENDPOINT, TEMPERATURE, TOP_P


async def llm_query(sess, query):
    assert 0.0 <= TEMPERATURE <= 1.0
    assert 0.0 <= TOP_P <= 1.0
    async with sess.post(
        API_ENDPOINT + API_KEY,
        json={
            "contents": [{"role": "USER", "parts": [{"text": query}]}],
            "generationConfig": {"temperature": TEMPERATURE, "topP": TOP_P},
        },
    ) as resp:
        return (await resp.json())["candidates"][0]["content"]["parts"][0]["text"]


async def llm_template_query(sess, template, passage):
    query = template["prompt"](passage).strip()
    response = await llm_query(sess, query)
    return template["extract"](response)
