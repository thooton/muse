import os
import configparser
import json


def load_configuration():
    CONFIG_FILE = "config.ini"

    config = configparser.ConfigParser()

    config_file_exists = os.path.exists(CONFIG_FILE)

    BOOL_TABLE = {"True": True, "False": False}

    if not config_file_exists:
        config["Gemini"] = {
            "API_KEYS": "[]",
            "API_ENDPOINT": "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=",
        }
        config["Parameters"] = {"TEMPERATURE": "1.0", "TOP_P": "0.99"}
        config["Misc"] = {
            "OUT_DIR": "./textbooks",
            "COUNT_PER_FILE": "1000",
            "BEGIN_INDEX": "0",
            "VERBOSE_EXCEPTIONS": "False",
        }
        with open("config.ini", "w") as configfile:
            config.write(configfile)

        print(
            "config.ini file not found. A default config.ini has been generated. Please update it with your API key and other configurations."
        )
        exit(1)

    config.read("config.ini")

    API_KEYS = json.loads(config.get("Gemini", "API_KEYS", fallback="[]"))
    API_ENDPOINT = config.get("Gemini", "API_ENDPOINT", fallback="")
    TEMPERATURE = float(config.get("Parameters", "TEMPERATURE", fallback="0.0"))
    TOP_P = float(config.get("Parameters", "TOP_P", fallback="0.0"))
    OUT_DIR = config.get("Misc", "OUT_DIR", fallback="")
    COUNT_PER_FILE = int(config.get("Misc", "COUNT_PER_FILE", fallback="0"))
    BEGIN_INDEX = int(config.get("Misc", "BEGIN_INDEX", fallback="0"))
    VERBOSE_EXCEPTIONS = BOOL_TABLE[
        config.get("Misc", "VERBOSE_EXCEPTIONS", fallback="False")
    ]
    DATASET_NAME = config.get("Misc", "DATASET_NAME", fallback="muse_textbooks")
    TEMPLATE_NAME = config.get("Misc", "TEMPLATE_NAME", fallback="textbooks")

    API_KEYS = os.getenv("API_KEYS", ";".join(API_KEYS)).split(";")
    API_ENDPOINT = os.getenv("API_ENDPOINT", API_ENDPOINT)
    TEMPERATURE = float(os.getenv("TEMPERATURE", TEMPERATURE))
    TOP_P = float(os.getenv("TOP_P", TOP_P))
    OUT_DIR = os.getenv("OUT_DIR", OUT_DIR)
    COUNT_PER_FILE = int(os.getenv("COUNT_PER_FILE", COUNT_PER_FILE))
    BEGIN_INDEX = int(os.getenv("BEGIN_INDEX", BEGIN_INDEX))
    VERBOSE_EXCEPTIONS = BOOL_TABLE[
        os.getenv("VERBOSE_EXCEPTIONS", str(VERBOSE_EXCEPTIONS))
    ]
    DATASET_NAME = os.getenv("DATASET_NAME", DATASET_NAME)

    if len(API_KEYS) == 0:
        print("API keys are empty. Please update config.ini with one or more API keys.")
        exit(1)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    return (
        API_KEYS,
        API_ENDPOINT,
        TEMPERATURE,
        TOP_P,
        OUT_DIR,
        COUNT_PER_FILE,
        BEGIN_INDEX,
        VERBOSE_EXCEPTIONS,
        DATASET_NAME,
        TEMPLATE_NAME,
    )


(
    API_KEYS,
    API_ENDPOINT,
    TEMPERATURE,
    TOP_P,
    OUT_DIR,
    COUNT_PER_FILE,
    BEGIN_INDEX,
    VERBOSE_EXCEPTIONS,
    DATASET_NAME,
    TEMPLATE_NAME,
) = load_configuration()
