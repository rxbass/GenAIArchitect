import json
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()


class KnowledgeGraphBuilder:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.username = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")

        if not all([self.uri, self.username, self.password]):
            raise ValueError(
                "NEO4J_URI, NEO4J_USERNAME and NEO4J_PASSWORD "
                "must be configured in .env"
            )

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password)
        )

    def close(self):
        self.driver.close()

    def create_constraints(self):
        queries = [
            """
            CREATE CONSTRAINT scheme_id_unique IF NOT EXISTS
            FOR (s:Scheme)
            REQUIRE s.scheme_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT department_name_unique IF NOT EXISTS
            FOR (d:Department)
            REQUIRE d.name IS UNIQUE
            """,
            """
            CREATE CONSTRAINT beneficiary_name_unique IF NOT EXISTS
            FOR (b:Beneficiary)
            REQUIRE b.name IS UNIQUE
            """,
            """
            CREATE CONSTRAINT benefit_type_name_unique IF NOT EXISTS
            FOR (bt:BenefitType)
            REQUIRE bt.name IS UNIQUE
            """,
            """
            CREATE CONSTRAINT sponsor_name_unique IF NOT EXISTS
            FOR (sp:Sponsor)
            REQUIRE sp.name IS UNIQUE
            """,
            """
            CREATE CONSTRAINT district_name_unique IF NOT EXISTS
            FOR (d:District)
            REQUIRE d.name IS UNIQUE
            """,
            """
            CREATE CONSTRAINT office_name_unique IF NOT EXISTS
            FOR (o:Office)
            REQUIRE o.name IS UNIQUE
            """,
        ]

        with self.driver.session() as session:
            for query in queries:
                session.run(query)

        print("Constraints created successfully.")

    def insert_scheme(self, scheme):
        query = """
        MERGE (s:Scheme {scheme_id: $scheme_id})
        SET
            s.name = $scheme_name,
            s.source_url = $source_url,
            s.sponsored_by = $sponsored_by,
            s.funding_pattern = $funding_pattern,
            s.description = $description,
            s.how_to_avail = $how_to_avail

        MERGE (d:Department {name: $department})
        MERGE (d)-[:OFFERS]->(s)

        WITH s

        FOREACH (beneficiary IN $beneficiaries |
            MERGE (b:Beneficiary {name: beneficiary})
            MERGE (s)-[:BENEFITS]->(b)
        )

        WITH s

        FOREACH (benefit_type IN $benefit_types |
            MERGE (bt:BenefitType {name: benefit_type})
            MERGE (s)-[:PROVIDES]->(bt)
        )

        WITH s

        FOREACH (sponsor IN
            CASE
                WHEN $sponsored_by <> ''
                THEN [$sponsored_by]
                ELSE []
            END |
            MERGE (sp:Sponsor {name: sponsor})
            MERGE (s)-[:SPONSORED_BY]->(sp)
        )

        WITH s

        FOREACH (district IN $districts |
            MERGE (dist:District {name: district})
            MERGE (s)-[:AVAILABLE_IN]->(dist)
        )

        WITH s

        FOREACH (office IN $application_offices |
            MERGE (o:Office {name: office})
            MERGE (s)-[:APPLIED_THROUGH]->(o)
        )
        """

        with self.driver.session() as session:
            session.run(
                query,
                scheme_id=scheme["scheme_id"],
                scheme_name=scheme["scheme_name"],
                source_url=scheme["source_url"],
                sponsored_by=scheme["sponsored_by"],
                funding_pattern=scheme["funding_pattern"],
                description=scheme["description"],
                how_to_avail=scheme["how_to_avail"],
                department=scheme["department"],
                beneficiaries=scheme["beneficiaries"],
                benefit_types=scheme["benefit_types"],
                districts=scheme["districts"],
                application_offices=scheme["application_offices"],
            )

    def build(self, schemes):
        self.create_constraints()

        for index, scheme in enumerate(schemes, start=1):
            self.insert_scheme(scheme)

            print(
                f"Inserted {index}/{len(schemes)}: "
                f"{scheme['scheme_name']}"
            )


def load_schemes():
    project_root = Path(__file__).resolve().parents[2]

    processed_file = (
        project_root
        / "data"
        / "processed"
        / "schemes.json"
    )

    if not processed_file.exists():
        raise FileNotFoundError(
            f"Processed corpus not found: {processed_file}"
        )

    with processed_file.open(
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def main():
    schemes = load_schemes()

    print(
        f"Loaded {len(schemes)} schemes "
        "from processed corpus."
    )

    builder = KnowledgeGraphBuilder()

    try:
        builder.build(schemes)

        print("\nKnowledge Graph construction completed.")

    finally:
        builder.close()


if __name__ == "__main__":
    main()