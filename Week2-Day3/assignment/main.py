from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv(r"D:\ai\GenAIArchitect\.env", override=True)

# Check required API keys
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is missing. Please add it to your .env file.")

if not os.getenv("LANGSMITH_API_KEY"):
    raise ValueError("LANGSMITH_API_KEY is missing. Please add it to your .env file.")


# --------------------------------------------------
# 2. Create the OpenAI chat model through LangChain
# --------------------------------------------------

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)


# --------------------------------------------------
# 3. Create a LangChain prompt template
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an aviation charter operations assistant.

        Analyze customer flight requests and create a concise
        operations brief.

        Identify:
        1. Flight summary
        2. Passenger requirements
        3. Missing information
        4. Operational attention items
        """
    ),
    (
        "human",
        """
        Analyze this charter flight request:

        Customer: {customer}
        Route: {route}
        Departure: {departure}
        Passengers: {passengers}
        Aircraft Preference: {aircraft}
        Special Requirements:
        {requirements}
        """
    )
])


# --------------------------------------------------
# 4. Output parser
# --------------------------------------------------

parser = StrOutputParser()


# --------------------------------------------------
# 5. Build a LangChain LCEL pipeline
# --------------------------------------------------

chain = prompt | model | parser


# --------------------------------------------------
# 6. Real-world aviation input
# --------------------------------------------------

flight_data = {
    "customer": "TechCorp International",
    "route": "Chennai → SFO",
    "departure": "25 September 2026, 08:30 IST",
    "passengers": "12",
    "aircraft": "Heavy Jet",
    "requirements": """
    - 2 passengers require vegetarian meals
    - One passenger has golf equipment
    - Customer requested Wi-Fi
    - Return flight required after 4 days
    - Return flight timing has not been confirmed
    """
}


# --------------------------------------------------
# 7. Invoke the LangChain pipeline
# --------------------------------------------------

print("=" * 60)
print("AVIATION CHARTER OPERATIONS ASSISTANT")
print("=" * 60)

print("\nInput:")
print(f"Customer: {flight_data['customer']}")
print(f"Route: {flight_data['route']}")
print(f"Departure: {flight_data['departure']}")
print(f"Passengers: {flight_data['passengers']}")

print("\nGenerating operations brief...\n")

result = chain.invoke(flight_data)


# --------------------------------------------------
# 8. Print the result
# --------------------------------------------------

print("Output:")
print(result)

print("\n" + "=" * 60)