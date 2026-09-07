import os

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()


class KnowledgeGraphQueries:

    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.username = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password)
        )

    def close(self):
        self.driver.close()

    def _run_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(
                query,
                parameters or {}
            )

            return [record.data() for record in result]

    def get_all_schemes(self):
        query = """
        MATCH (s:Scheme)
        RETURN
            s.scheme_id AS scheme_id,
            s.name AS scheme_name,
            s.source_url AS source_url
        ORDER BY s.scheme_id
        """

        return self._run_query(query)

    def get_schemes_by_beneficiary(self, beneficiary):
        query = """
        MATCH (s:Scheme)-[:BENEFITS]->(b:Beneficiary)
        WHERE toLower(b.name) CONTAINS toLower($beneficiary)
        RETURN
            s.scheme_id AS scheme_id,
            s.name AS scheme_name,
            b.name AS beneficiary
        ORDER BY s.name
        """

        return self._run_query(
            query,
            {"beneficiary": beneficiary}
        )

    def get_schemes_by_district(self, district):
        query = """
        MATCH (s:Scheme)-[:AVAILABLE_IN]->(d:District)
        WHERE toLower(d.name) CONTAINS toLower($district)
        RETURN
            s.scheme_id AS scheme_id,
            s.name AS scheme_name,
            d.name AS district
        ORDER BY s.name
        """

        return self._run_query(
            query,
            {"district": district}
        )

    def get_schemes_by_benefit(self, benefit_type):
        query = """
        MATCH (s:Scheme)-[:PROVIDES]->(b:BenefitType)
        WHERE toLower(b.name) CONTAINS toLower($benefit_type)
        RETURN
            s.scheme_id AS scheme_id,
            s.name AS scheme_name,
            b.name AS benefit_type
        ORDER BY s.name
        """

        return self._run_query(
            query,
            {"benefit_type": benefit_type}
        )

    def get_schemes_by_department(self, department):
        query = """
        MATCH (d:Department)-[:OFFERS]->(s:Scheme)
        WHERE toLower(d.name) CONTAINS toLower($department)
        RETURN
            s.scheme_id AS scheme_id,
            s.name AS scheme_name,
            d.name AS department
        ORDER BY s.name
        """

        return self._run_query(
            query,
            {"department": department}
        )

    def get_scheme_details(self, scheme_name):
        query = """
        MATCH (s:Scheme)
        WHERE toLower(s.name) CONTAINS toLower($scheme_name)

        OPTIONAL MATCH (d:Department)-[:OFFERS]->(s)
        OPTIONAL MATCH (s)-[:BENEFITS]->(b:Beneficiary)
        OPTIONAL MATCH (s)-[:PROVIDES]->(bt:BenefitType)
        OPTIONAL MATCH (s)-[:SPONSORED_BY]->(sp:Sponsor)
        OPTIONAL MATCH (s)-[:AVAILABLE_IN]->(dist:District)
        OPTIONAL MATCH (s)-[:APPLIED_THROUGH]->(o:Office)

        RETURN
            s.scheme_id AS scheme_id,
            s.name AS scheme_name,
            d.name AS department,
            s.sponsored_by AS sponsored_by,
            s.funding_pattern AS funding_pattern,
            s.description AS description,
            s.how_to_avail AS how_to_avail,
            s.source_url AS source_url,
            collect(DISTINCT b.name) AS beneficiaries,
            collect(DISTINCT bt.name) AS benefit_types,
            collect(DISTINCT sp.name) AS sponsors,
            collect(DISTINCT dist.name) AS districts,
            collect(DISTINCT o.name) AS application_offices
        """

        return self._run_query(
            query,
            {"scheme_name": scheme_name}
        )


def main():

    kg = KnowledgeGraphQueries()

    try:

        print("\n--- ALL SCHEMES ---")

        schemes = kg.get_all_schemes()

        print(f"Total schemes: {len(schemes)}")

        for scheme in schemes[:5]:
            print(scheme)

        print("\n--- FARMER SCHEMES ---")

        farmer_schemes = kg.get_schemes_by_beneficiary(
            "Farmers"
        )

        for scheme in farmer_schemes[:10]:
            print(scheme)

        print("\n--- COIMBATORE SCHEMES ---")

        coimbatore_schemes = kg.get_schemes_by_district(
            "Coimbatore"
        )

        for scheme in coimbatore_schemes[:10]:
            print(scheme)

        print("\n--- GRANT SCHEMES ---")

        grant_schemes = kg.get_schemes_by_benefit(
            "Grants"
        )

        for scheme in grant_schemes[:10]:
            print(scheme)

        print("\n--- SCHEME DETAILS ---")

        details = kg.get_scheme_details(
            "Training to Farmers"
        )

        for item in details:
            print(item)

    finally:

        kg.close()


if __name__ == "__main__":
    main()