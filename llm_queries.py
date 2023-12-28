from config import API_ENDPOINT, TEMPERATURE, TOP_P


async def llm_query(sess, query, api_key):
    assert 0.0 <= TEMPERATURE <= 1.0
    assert 0.0 <= TOP_P <= 1.0
    async with sess.post(
        API_ENDPOINT + api_key,
        json={
            "contents": [{"role": "USER", "parts": [{"text": query}]}],
            "generationConfig": {"temperature": TEMPERATURE, "topP": TOP_P},
        },
    ) as resp:
        return (await resp.json())["candidates"][0]["content"]["parts"][0]["text"]


async def llm_template_query(sess, template, passage, api_key):
    try:
        query = template["prompt"](passage).strip()
        response = await llm_query(sess, query, api_key)
        return ("ok", template["extract"](response))
    except Exception as exc:
        return ("err", {"exc": exc, "api_key": api_key})
