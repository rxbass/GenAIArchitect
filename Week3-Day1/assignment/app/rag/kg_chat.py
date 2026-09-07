import re
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph


from app.config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
)
from app.guardrails.retrieval_guard import validate_retrieval

class KnowledgeGraphChat:

    def __init__(self):

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )

        self.graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
        )

    # =========================================================
    # Generate Cypher
    # =========================================================

    def generate_cypher(
        self,
        question: str,
    ) -> str:

        prompt = f"""
You are an expert Neo4j Cypher query generator.

Convert the user's question into ONE valid READ-ONLY Cypher
query.

============================================================
SCHEMA
============================================================

Nodes:

(:Department)
(:Scheme)
(:Beneficiary)
(:BenefitType)
(:Sponsor)
(:District)
(:Office)

Scheme properties:

scheme_id
name
source_url
sponsored_by
funding_pattern
description
how_to_avail

Relationships:

(:Department)-[:OFFERS]->(:Scheme)

(:Scheme)-[:BENEFITS]->(:Beneficiary)

(:Scheme)-[:PROVIDES]->(:BenefitType)

(:Scheme)-[:SPONSORED_BY]->(:Sponsor)

(:Scheme)-[:AVAILABLE_IN]->(:District)

(:Scheme)-[:APPLIED_THROUGH]->(:Office)

============================================================
RULES
============================================================

1. READ-ONLY Cypher only.

2. Must contain MATCH and RETURN.

3. Never use:
CREATE
DELETE
DETACH
SET
REMOVE
MERGE
DROP
ALTER
LOAD CSV
CALL
FOREACH
GRANT
DENY
REVOKE

4. Never reference a variable that was not introduced by
MATCH or OPTIONAL MATCH.

5. Department relationship direction:

(d:Department)-[:OFFERS]->(s:Scheme)

6. Beneficiary:

(s:Scheme)-[:BENEFITS]->(b:Beneficiary)

7. Benefit:

(s:Scheme)-[:PROVIDES]->(bt:BenefitType)

8. Sponsor:

(s:Scheme)-[:SPONSORED_BY]->(sp:Sponsor)

9. District:

(s:Scheme)-[:AVAILABLE_IN]->(di:District)

10. Office:

(s:Scheme)-[:APPLIED_THROUGH]->(o:Office)

11. For list questions, return ONE ROW PER SCHEME.

12. Do not use collect() unless the corresponding variable
has first been matched.

13. For a simple list query, return only the fields needed
to answer the question.

14. For a detailed scheme question, use OPTIONAL MATCH and
collect(DISTINCT ...) for multi-valued relationships.

15. Do not use Markdown fences.

============================================================
EXAMPLES
============================================================

Question:
What schemes are available for farmers?

Query:

MATCH (s:Scheme)-[:BENEFITS]->(b:Beneficiary)
WHERE toLower(b.name) CONTAINS 'farmer'
RETURN
    s.scheme_id AS scheme_id,
    s.name AS scheme_name,
    s.source_url AS source_url,
    s.description AS description,
    s.how_to_avail AS how_to_avail
ORDER BY s.scheme_id


Question:
What schemes are available in Coimbatore?

Query:

MATCH (s:Scheme)-[:AVAILABLE_IN]->(di:District)
WHERE toLower(di.name) = toLower('Coimbatore')
RETURN
    s.scheme_id AS scheme_id,
    s.name AS scheme_name,
    s.source_url AS source_url,
    s.description AS description,
    s.how_to_avail AS how_to_avail
ORDER BY s.scheme_id


Question:
Which schemes provide grants?

Query:

MATCH (s:Scheme)-[:PROVIDES]->(bt:BenefitType)
WHERE toLower(bt.name) CONTAINS 'grant'
RETURN
    s.scheme_id AS scheme_id,
    s.name AS scheme_name,
    s.source_url AS source_url,
    s.description AS description,
    s.how_to_avail AS how_to_avail
ORDER BY s.scheme_id


Question:
Which department offers Training to Farmers?

Query:

MATCH (d:Department)-[:OFFERS]->(s:Scheme)
WHERE toLower(s.name) = toLower('Training to Farmers')
RETURN
    s.scheme_id AS scheme_id,
    s.name AS scheme_name,
    collect(DISTINCT d.name) AS departments
ORDER BY s.scheme_id


Question:
Where can I apply for Training to Farmers?

Query:

MATCH (s:Scheme)
WHERE toLower(s.name) = toLower('Training to Farmers')

OPTIONAL MATCH
    (s)-[:APPLIED_THROUGH]->(o:Office)

RETURN
    s.scheme_id AS scheme_id,
    s.name AS scheme_name,
    collect(DISTINCT o.name) AS offices
ORDER BY s.scheme_id


Question:
Tell me about Training to Farmers?

Query:

MATCH (s:Scheme)
WHERE toLower(s.name) = toLower('Training to Farmers')

OPTIONAL MATCH
    (d:Department)-[:OFFERS]->(s)

OPTIONAL MATCH
    (s)-[:BENEFITS]->(b:Beneficiary)

OPTIONAL MATCH
    (s)-[:PROVIDES]->(bt:BenefitType)

OPTIONAL MATCH
    (s)-[:SPONSORED_BY]->(sp:Sponsor)

OPTIONAL MATCH
    (s)-[:AVAILABLE_IN]->(di:District)

OPTIONAL MATCH
    (s)-[:APPLIED_THROUGH]->(o:Office)

RETURN
    s.scheme_id AS scheme_id,
    s.name AS scheme_name,
    s.source_url AS source_url,
    s.description AS description,
    s.how_to_avail AS how_to_avail,
    collect(DISTINCT d.name) AS departments,
    collect(DISTINCT b.name) AS beneficiaries,
    collect(DISTINCT bt.name) AS benefits,
    collect(DISTINCT sp.name) AS sponsors,
    collect(DISTINCT di.name) AS districts,
    collect(DISTINCT o.name) AS offices
ORDER BY s.scheme_id

============================================================
USER QUESTION
============================================================

{question}

Return ONLY Cypher.
"""

        response = self.llm.invoke(prompt)

        content = response.content

        if isinstance(content, list):
            content = "".join(
                str(item)
                for item in content
            )

        return self.clean_cypher(
            content
        )

    # =========================================================
    # Clean Cypher
    # =========================================================

    def clean_cypher(
        self,
        cypher: str,
    ) -> str:

        cypher = cypher.strip()

        cypher = re.sub(
            r"^```(?:cypher)?\s*",
            "",
            cypher,
            flags=re.IGNORECASE,
        )

        cypher = re.sub(
            r"\s*```$",
            "",
            cypher,
        )

        return cypher.strip()

    # =========================================================
    # Validate Cypher
    # =========================================================

    def validate_cypher(
        self,
        cypher: str,
    ) -> bool:

        normalized = re.sub(
            r"\s+",
            " ",
            cypher.strip(),
        ).upper()

        forbidden_patterns = [
            r"\bCREATE\b",
            r"\bDELETE\b",
            r"\bDETACH\b",
            r"\bSET\b",
            r"\bREMOVE\b",
            r"\bMERGE\b",
            r"\bDROP\b",
            r"\bALTER\b",
            r"\bLOAD\s+CSV\b",
            r"\bCALL\b",
            r"\bFOREACH\b",

            # Only block actual privilege commands.
            # Do NOT block the data value "grant".
            r"\bGRANT\s+(ROLE|PRIVILEGE|ACCESS)\b",
            r"\bDENY\s+(ROLE|PRIVILEGE|ACCESS)\b",
            r"\bREVOKE\s+(ROLE|PRIVILEGE|ACCESS)\b",
        ]

        for pattern in forbidden_patterns:

            if re.search(
                pattern,
                normalized,
            ):

                print(
                    f"Blocked Cypher keyword/pattern: {pattern}"
                )

                return False

        if not re.search(
            r"\bMATCH\b",
            normalized,
        ):

            print(
                "Cypher validation failed: MATCH missing"
            )

            return False

        if not re.search(
            r"\bRETURN\b",
            normalized,
        ):

            print(
                "Cypher validation failed: RETURN missing"
            )

            return False

        if "```" in cypher:

            print(
                "Cypher validation failed: Markdown fence"
            )

            return False

        return True

    # =========================================================
    # Execute
    # =========================================================

    def execute_cypher(
        self,
        cypher: str,
    ) -> list[dict[str, Any]]:

        return self.graph.query(
            cypher
        )

    # =========================================================
    # Deterministic result formatting
    # =========================================================

    def format_results(
        self,
        results: list[dict[str, Any]],
    ) -> str:

        """
        Convert every graph result into a deterministic text
        representation.

        This ensures the final LLM receives every retrieved
        record and makes omissions easier to detect.
        """

        lines = []

        for index, result in enumerate(
            results,
            start=1,
        ):

            lines.append(
                f"RESULT {index}:"
            )

            for key, value in result.items():

                if value is None:
                    continue

                if isinstance(value, list):

                    clean_values = [
                        str(item)
                        for item in value
                        if item is not None
                    ]

                    if clean_values:
                        lines.append(
                            f"{key}: "
                            + ", ".join(
                                clean_values
                            )
                        )

                else:

                    lines.append(
                        f"{key}: {value}"
                    )

            lines.append("")

        return "\n".join(lines)

    # =========================================================
    # Generate final answer
    # =========================================================

    def generate_answer(
        self,
        question: str,
        cypher: str,
        results: list[dict[str, Any]],
    ) -> str:

        if not results:

            return (
                "I could not find any matching Tamil Nadu "
                "Government scheme in the Knowledge Graph."
            )

        formatted_results = self.format_results(
            results
        )

        prompt = f"""
You are a Knowledge Graph answer generator.

Answer the user's question using ONLY the retrieved graph
results.

USER QUESTION:

{question}

RETRIEVED GRAPH RESULTS:

{formatted_results}

STRICT RULES:

1. Use ONLY the retrieved graph results.

2. Do not use outside knowledge.

3. Do not invent facts.

4. Every retrieved scheme must be preserved when the question
asks for a list of schemes.

5. Never omit a retrieved scheme simply because there are many.

6. Never add a scheme that is not present in the results.

7. Never create duplicate schemes.

8. If a field is missing or empty, do not invent a value.

9. If source_url is present, include it where useful.

10. For list questions, provide a numbered list.

11. For a single scheme question, provide a concise summary.

12. Do not mention Cypher or internal implementation details.

13. Keep the answer clear and concise.

Answer directly.
"""

        response = self.llm.invoke(
            prompt
        )

        answer = response.content

        if isinstance(answer, list):

            answer = "".join(
                str(item)
                for item in answer
            )

        return answer.strip()

    # =========================================================
    # Answer question
    # =========================================================

    def answer_question(
        self,
        question: str,
    ) -> str:

        print("\n" + "=" * 70)
        print("QUESTION")
        print("=" * 70)

        print(question)

        # -----------------------------------------------------
        # Generate Cypher
        # -----------------------------------------------------

        cypher = self.generate_cypher(
            question
        )

        print("\n" + "=" * 70)
        print("GENERATED CYPHER")
        print("=" * 70)

        print(cypher)

        # -----------------------------------------------------
        # Validate
        # -----------------------------------------------------

        if not self.validate_cypher(
            cypher
        ):

            print(
                "\nCypher rejected by validation."
            )

            return (
                "I could not safely execute the generated "
                "Knowledge Graph query."
            )

        # -----------------------------------------------------
        # Execute
        # -----------------------------------------------------

        try:

            results = self.execute_cypher(
                cypher
            )

        except Exception as exc:

            print(
                "\n" + "=" * 70
            )

            print(
                "NEO4J EXECUTION ERROR"
            )

            print(
                "=" * 70
            )

            print(exc)

            return (
                "I could not execute the Knowledge Graph query."
            )

        # -----------------------------------------------------
        # Retrieval Guardrail
        # -----------------------------------------------------

        retrieval_allowed, retrieval_message = validate_retrieval(
            results
        )

        if not retrieval_allowed:

            print(
                "\nRetrieval guardrail rejected the graph results."
            )

            print(
                retrieval_message
            )

            return (
                "I could not safely process the information "
                "retrieved from the Knowledge Graph."
            )


        # -----------------------------------------------------
        # Graph results
        # -----------------------------------------------------

        print(
            "\n" + "=" * 70
        )

        print(
            "GRAPH RESULTS"
        )

        print(
            "=" * 70
        )

        for result in results:

            print(result)


        # -----------------------------------------------------
        # Answer
        # -----------------------------------------------------

        answer = self.generate_answer(
            question=question,
            cypher=cypher,
            results=results,
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "FINAL ANSWER"
        )

        print(
            "=" * 70
        )

        print(answer)

        return answer


# =============================================================
# CLI
# =============================================================

def main():

    chatbot = KnowledgeGraphChat()

    print(
        "\nTamil Nadu Government Scheme "
        "Knowledge Graph Chatbot"
    )

    print(
        "Type 'exit' to quit.\n"
    )

    while True:

        question = input(
            "You: "
        ).strip()

        if question.lower() in {
            "exit",
            "quit",
        }:

            print(
                "Goodbye!"
            )

            break

        if not question:
            continue

        chatbot.answer_question(
            question
        )


if __name__ == "__main__":
    main()