import os
import configparser


def load_configuration():
    CONFIG_FILE = "config.ini"

    config = configparser.ConfigParser()

    config_file_exists = os.path.exists(CONFIG_FILE)

    API_KEY = os.getenv("API_KEY", "")
    API_ENDPOINT = os.getenv(
        "API_ENDPOINT",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=",
    )
    TEMPERATURE = float(os.getenv("TEMPERATURE", 1.0))
    TOP_P = float(os.getenv("TOP_P", 0.99))
    OUT_DIR = os.getenv("OUT_DIR", "./textbooks")
    COUNT_PER_FILE = int(os.getenv("COUNT_PER_FILE", 1000))

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    if config_file_exists and "" in (API_KEY, API_ENDPOINT):
        config.read("config.ini")
        API_KEY = config.get("Gemini", "API_KEY", fallback="")
        API_ENDPOINT = config.get(
            "Gemini",
            "API_ENDPOINT",
            fallback="https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=",
        )

        if API_KEY == "":
            print("API key is empty. Please update config.ini with your API key.")
            exit(1)

    else:
        config["Gemini"] = {
            "API_KEY": API_KEY,
            "API_ENDPOINT": API_ENDPOINT,
        }
        config["Parameters"] = {"TEMPERATURE": str(TEMPERATURE), "TOP_P": str(TOP_P)}
        config["Misc"] = {
            "OUT_DIR": OUT_DIR,
            "COUNT_PER_FILE": str(COUNT_PER_FILE),
        }
        with open("config.ini", "w") as configfile:
            config.write(configfile)

        print(
            "config.ini file not found. A default config.ini has been generated. Please update it with your API key and other configurations."
        )
        exit(1)

    return API_KEY, API_ENDPOINT, TEMPERATURE, TOP_P, OUT_DIR, COUNT_PER_FILE


(
    API_KEY,
    API_ENDPOINT,
    TEMPERATURE,
    TOP_P,
    OUT_DIR,
    COUNT_PER_FILE,
) = load_configuration()
