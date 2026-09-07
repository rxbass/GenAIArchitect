# app/ingestion/processor.py

import json
import os
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


FIELD_NAMES = [
    "Concerned Department:",
    "Concerned District:",
    "Organisation Name:",
    "Department Scheme Details:",
    "Scheme Title/Name:",
    "Associated Scheme:",
    "Sponsered By:",
    "Funding Pattern:",
    "Beneficiaries:",
    "Types of Benefits:",
    "Eligibility criteria:",
    "Income:",
    "Age From:",
    "Age To:",
    "Community:",
    "How To avail:",
    "The application is to be submitted to:",
    "Application is to be submitted to:",
    "Validity of the Scheme:",
    "Introduced On:",
    "Description:",
    "Scheme Type:",
    "Uploaded File:",
]


def clean_text(text: str) -> str:

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        # Remove obvious website chrome
        if line in {
            "Screen Reader Access",
            "Accessibility Menu",
            "Text To Speech",
            "Bigger Text",
            "Small Text",
            "Line Height",
            "Highlight Links",
            "Text Spacing",
            "Cursor",
            "Light-Dark",
            "Invert Colors",
            "Reset All Settings",
            "Accessibility Options",
            "Skip To Main Content",
            "News & Events",
            "Press Releases",
            "Photo Gallery",
            "Video Gallery",
            "Interact with Govt",
            "Polls",
            "Discussion Forum",
            "Others",
            "What's New",
            "Help line Numbers",
            "Holidays",
            "Sitemap",
            "Feedback",
        }:
            continue

        if line.startswith(
            "Your browser does not support"
        ):
            continue

        if line.startswith(
            "Content owned and Maintained"
        ):
            break

        lines.append(line)

    return "\n".join(lines)


def parse_fields(text: str) -> dict:

    lines = text.splitlines()

    data = {}

    current_field = None
    current_value = []

    field_lookup = {
        field.lower(): field
        for field in FIELD_NAMES
    }

    def save_current():

        nonlocal current_field

        if current_field:

            value = " ".join(
                current_value
            ).strip()

            # Ignore empty values
            if value:
                data[current_field] = value

    for raw_line in lines:

        normalized = raw_line.strip()

        if not normalized:
            continue

        # Ignore website section/header
        if normalized.lower() == (
            "department scheme details:"
        ):

            save_current()

            current_field = None
            current_value = []

            continue

        # Check whether this line is a known field
        matched_field = field_lookup.get(
            normalized.lower()
        )

        if matched_field:

            save_current()

            current_field = matched_field
            current_value = []

            continue

        # Collect value for current field
        if current_field:

            current_value.append(normalized)

    # Save final field
    save_current()

    return data


def add_field(parts, label, value):

    if value and value.strip():

        parts.append(
            f"{label}: {value.strip()}"
        )


def create_document(source, parsed):

    scheme_name = parsed.get(
        "Scheme Title/Name:",
        source["name"]
    ).strip()

    department = parsed.get(
        "Concerned Department:",
        ""
    ).strip()

    parts = []

    add_field(
        parts,
        "Scheme Name",
        scheme_name
    )

    add_field(
        parts,
        "Department",
        department
    )

    add_field(
        parts,
        "Concerned District",
        parsed.get(
            "Concerned District:"
        )
    )

    # Protect against website chrome being
    # incorrectly captured as an organisation.
    organisation = parsed.get(
        "Organisation Name:"
    )

    if (
        organisation
        and organisation.strip().lower()
        != "department scheme details:"
    ):

        add_field(
            parts,
            "Organisation",
            organisation
        )

    add_field(
        parts,
        "Associated Scheme",
        parsed.get(
            "Associated Scheme:"
        )
    )

    add_field(
        parts,
        "Sponsored By",
        parsed.get(
            "Sponsered By:"
        )
    )

    add_field(
        parts,
        "Funding Pattern",
        parsed.get(
            "Funding Pattern:"
        )
    )

    add_field(
        parts,
        "Beneficiaries",
        parsed.get(
            "Beneficiaries:"
        )
    )

    add_field(
        parts,
        "Types of Benefits",
        parsed.get(
            "Types of Benefits:"
        )
    )

    # Eligibility
    eligibility = []

    for field, label in [
        ("Income:", "Income"),
        ("Age From:", "Age From"),
        ("Age To:", "Age To"),
        ("Community:", "Community"),
    ]:

        value = parsed.get(field)

        if value and value.strip():

            eligibility.append(
                f"{label}: {value.strip()}"
            )

    if eligibility:

        parts.append(
            "Eligibility:\n"
            + "\n".join(eligibility)
        )

    add_field(
        parts,
        "How To Avail",
        parsed.get(
            "How To avail:"
        )
    )

    application_to = (
        parsed.get(
            "The application is to be submitted to:"
        )
        or
        parsed.get(
            "Application is to be submitted to:"
        )
        or
        ""
    ).strip()

    add_field(
        parts,
        "Application Submitted To",
        application_to
    )

    add_field(
        parts,
        "Validity",
        parsed.get(
            "Validity of the Scheme:"
        )
    )

    add_field(
        parts,
        "Introduced On",
        parsed.get(
            "Introduced On:"
        )
    )

    add_field(
        parts,
        "Description",
        parsed.get(
            "Description:"
        )
    )

    add_field(
        parts,
        "Scheme Type",
        parsed.get(
            "Scheme Type:"
        )
    )

    content = "\n\n".join(parts)

    return Document(
        page_content=content,
        metadata={
            "scheme_id": source["id"],
            "scheme_name": scheme_name,
            "department": department,
            "source_url": source["url"],
        }
    )


def build_structured_scheme(source, parsed):

    scheme_name = parsed.get(
        "Scheme Title/Name:",
        source["name"]
    ).strip()

    department = parsed.get(
        "Concerned Department:",
        ""
    ).strip()

    sponsored_by = parsed.get(
        "Sponsered By:",
        ""
    ).strip()

    funding_pattern = parsed.get(
        "Funding Pattern:",
        ""
    ).strip()

    beneficiaries = parsed.get(
        "Beneficiaries:",
        ""
    ).strip()

    benefit_types = parsed.get(
        "Types of Benefits:",
        ""
    ).strip()

    description = parsed.get(
        "Description:",
        ""
    ).strip()

    application_to = (
        parsed.get(
            "The application is to be submitted to:"
        )
        or
        parsed.get(
            "Application is to be submitted to:"
        )
        or
        ""
    ).strip()

    # -------------------------------------------------
    # Extract application offices
    # -------------------------------------------------

    application_offices = []

    if application_to:

        office_patterns = [
            r"Assistant Agricultural officer at the Village Level",
            r"Agricultural Officer / Deputy Agricultural Officer at the Block Level",
            r"The Assistant Director of Agriculture at the Block Level",
            r"The Joint Director of Agriculture at the District Level",
        ]

        for pattern in office_patterns:

            match = re.search(
                pattern,
                application_to,
                flags=re.IGNORECASE
            )

            if match:

                application_offices.append(
                    match.group(0).strip()
                )

        # If no known office pattern matched,
        # preserve the complete source value.
        if not application_offices:

            application_offices = [
                application_to
            ]

    # -------------------------------------------------
    # Extract districts from description
    # -------------------------------------------------

    districts = []

    known_districts = [
        "Ariyalur",
        "Chengalpattu",
        "Chennai",
        "Coimbatore",
        "Cuddalore",
        "Dharmapuri",
        "Dindigul",
        "Erode",
        "Kallakurichi",
        "Kancheepuram",
        "Karur",
        "Krishnagiri",
        "Madurai",
        "Mayiladuthurai",
        "Nagapattinam",
        "Namakkal",
        "Perambalur",
        "Pudukkottai",
        "Ramanathapuram",
        "Ranipet",
        "Salem",
        "Sivaganga",
        "Tenkasi",
        "Thanjavur",
        "Theni",
        "Thoothukudi",
        "Trichy",
        "Tiruchirappalli",
        "Tirunelveli",
        "Tirupathur",
        "Tiruppur",
        "Tiruvallur",
        "Tiruvarur",
        "Vellore",
        "Viluppuram",
        "Villupuram",
        "Virudhunagar",
    ]

    description_lower = description.lower()

    for district in known_districts:

        if district.lower() in description_lower:

            districts.append(district)

    return {
        "scheme_id": source["id"],
        "scheme_name": scheme_name,
        "department": department,
        "source_url": source["url"],
        "sponsored_by": sponsored_by,
        "funding_pattern": funding_pattern,
        "beneficiaries": (
            [beneficiaries]
            if beneficiaries
            else []
        ),
        "benefit_types": (
            [benefit_types]
            if benefit_types
            else []
        ),
        "districts": districts,
        "application_offices": application_offices,
        "description": description,
        "eligibility": {
            "income": parsed.get(
                "Income:",
                ""
            ).strip(),
            "age_from": parsed.get(
                "Age From:",
                ""
            ).strip(),
            "age_to": parsed.get(
                "Age To:",
                ""
            ).strip(),
            "community": parsed.get(
                "Community:",
                ""
            ).strip(),
        },
        "how_to_avail": parsed.get(
            "How To avail:",
            ""
        ).strip(),
    }


def process_corpus():

    with open(
        "data/raw/schemes.json",
        "r",
        encoding="utf-8"
    ) as f:

        sources = json.load(f)

    documents = []
    structured_schemes = []

    for source in sources:

        if source["status"] != "success":
            continue

        cleaned = clean_text(
            source["text"]
        )

        parsed = parse_fields(
            cleaned
        )

        structured_scheme = (
            build_structured_scheme(
                source,
                parsed
            )
        )

        structured_schemes.append(
            structured_scheme
        )

        document = create_document(
            source,
            parsed
        )

        documents.append(
            document
        )

    # Save structured dataset
    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    with open(
        "data/processed/schemes.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            structured_schemes,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        "Saved structured schemes to: "
        "data/processed/schemes.json"
    )

    return documents


def chunk_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=300,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
        ]
    )

    return splitter.split_documents(
        documents
    )


if __name__ == "__main__":

    documents = process_corpus()

    print(
        f"Processed documents: {len(documents)}"
    )

    chunks = chunk_documents(
        documents
    )

    print(
        f"Generated chunks: {len(chunks)}"
    )

    for chunk in chunks[:2]:

        print(
            "\n========== CHUNK =========="
        )

        print(
            chunk.page_content
        )

        print(
            "\nMETADATA:"
        )

        print(
            chunk.metadata
        )

