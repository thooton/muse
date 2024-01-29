import os
from tqdm import tqdm
import aiohttp
import asyncio
import secrets
import json
import huggingface_hub
from config import (
    API_KEYS,
    OUT_DIR,
    COUNT_PER_FILE,
    BEGIN_INDEX,
    VERBOSE_EXCEPTIONS,
    DATASET_NAME,
    TEMPLATE_NAME,
)
from data_processing import (
    TEMPLATE_GROUPS,
)
from llm_queries import llm_template_query
import traceback


def exc_fmt(exc):
    if VERBOSE_EXCEPTIONS:
        return "\n".join(traceback.format_exception(exc)).strip()
    else:
        return str(repr(exc))


async def main():
    huggingface_hub.login(new_session=False)
    hf_api = huggingface_hub.HfApi()

    hf_user = hf_api.whoami()["name"]
    repo_id = f"{hf_user}/{DATASET_NAME}"
    templates, dataset_map = TEMPLATE_GROUPS[TEMPLATE_NAME]

    hf_api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    # Iterate over each key, if it is a list make it iter otherwise skip
    dataset_map = {
        k: v if isinstance(v, list) else iter([v]) for k, v in dataset_map.items()
    }

    # Make each one inifinite loop
    import itertools

    dataset_map = {k: itertools.cycle(v) for k, v in dataset_map.items()}

    sess = aiohttp.ClientSession()

    tasks = set()
    lines = 0

    try:
        with open(os.path.join(OUT_DIR, "cur.jsonl"), "rb") as f:
            lines = len(f.read().decode("utf-8").split("\n")) - 1
    except Exception:
        pass

    pbar = tqdm(initial=lines, total=COUNT_PER_FILE)
    outfile = open(os.path.join(OUT_DIR, "cur.jsonl"), "ab")

    while True:
        for api_key in API_KEYS:
            # Get a random template
            template = templates[secrets.randbits(64) % len(templates)]

            # Get the required input for the template
            dataset_type = template["dataset"]
            passage = next(dataset_map[dataset_type])

            tasks.add(
                asyncio.create_task(
                    llm_template_query(sess, template, passage, api_key)
                )
            )

        new_tasks = set()

        for task in tasks:
            if not task.done():
                new_tasks.add(task)
                continue

            try:
                status, result = task.result()
            except Exception as exc:
                tqdm.write(f"unknown error: {exc_fmt(exc)}")
                continue

            if status == "err":
                api_key = result["api_key"]
                exc = result["exc"]
                tqdm.write(f"error in {api_key}: {exc_fmt(exc)}")
                continue

            if len(result) == 0:
                continue

            outfile.write((json.dumps({"text": result}) + "\n").encode("utf-8"))
            lines += 1
            pbar.update(1)

        tasks = new_tasks

        if lines >= COUNT_PER_FILE:
            outfile.close()
            i = BEGIN_INDEX

            while os.path.exists(os.path.join(OUT_DIR, f"{i}.jsonl")):
                i += 1

            os.rename(
                os.path.join(OUT_DIR, "cur.jsonl"), os.path.join(OUT_DIR, f"{i}.jsonl")
            )

            while True:
                try:
                    hf_api.upload_file(
                        path_or_fileobj=os.path.join(OUT_DIR, f"{i}.jsonl"),
                        path_in_repo=f"{i}.jsonl",
                        repo_id=repo_id,
                        repo_type="dataset",
                    )
                    break
                except Exception as exc:
                    tqdm.write(f"can't upload: {exc_fmt(exc)}")
                    await asyncio.sleep(1)

            outfile = open(os.path.join(OUT_DIR, "cur.jsonl"), "ab")
            lines = 0
            pbar.reset()

        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
