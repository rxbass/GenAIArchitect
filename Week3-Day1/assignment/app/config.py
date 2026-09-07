# app/config.py

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

ALLOWED_DOMAINS = {
    "tn.gov.in",
    "tnrd.tn.gov.in",
    "cra.tn.gov.in",
    "scd.tn.gov.in",
}