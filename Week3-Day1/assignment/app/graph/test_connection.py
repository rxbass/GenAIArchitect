import os

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()


NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def main():

    print("Neo4j URI:", NEO4J_URI)
    print("Neo4j username:", NEO4J_USERNAME)

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(
            NEO4J_USERNAME,
            NEO4J_PASSWORD
        )
    )

    try:

        driver.verify_connectivity()

        print("\nNeo4j connection successful!")

        with driver.session() as session:

            result = session.run(
                "RETURN 'Knowledge Graph Ready' AS message"
            )

            record = result.single()

            print(
                "Neo4j response:",
                record["message"]
            )

    except Exception as e:

        print("\nNeo4j connection failed!")
        print(type(e).__name__)
        print(e)

    finally:

        driver.close()


if __name__ == "__main__":
    main()