# Aviation Charter Operations Brief Generator

A beginner-to-medium level **LangChain assignment** demonstrating a
real-world GenAI application using **Python, LangChain, OpenAI, and
LangSmith**.

The application accepts an aviation charter flight request and uses an
OpenAI chat model **through LangChain** to generate a concise **Flight
Operations Brief**. Every LangChain execution is traced in **LangSmith**
for observability.

------------------------------------------------------------------------

## 1. Project Overview

In aviation charter operations, customer requests can contain flight
details, passenger requirements, aircraft preferences, special services,
and information that has not yet been confirmed.

This prototype uses an LLM to analyze the request and produce an
operational summary.

### Example Input

``` text
Customer: ABC Corporation
Route: Chennai → Dubai
Departure: 15 September 2026, 08:30 IST
Passengers: 12
Aircraft Preference: Heavy Jet

Special Requirements:
- 2 passengers require vegetarian meals
- One passenger has golf equipment
- Customer requested Wi-Fi
- Return flight required after 4 days
- Return flight timing has not been confirmed
```

### Example Output

``` text
Operations Brief

Flight Summary:
Chennai → Dubai
Departure: 15 September 2026, 08:30 IST
Passengers: 12
Aircraft: Heavy Jet

Passenger Requirements:
- Vegetarian meals for 2 passengers
- Golf equipment
- Wi-Fi

Missing Information:
- Return flight timing

Operational Attention:
- Confirm return flight timing
- Confirm golf equipment handling
- Confirm Wi-Fi availability
- Confirm catering requirements
```

------------------------------------------------------------------------

## 2. What This Assignment Demonstrates

-   Python virtual environment setup
-   LangChain framework
-   LangChain Expression Language (LCEL)
-   `ChatPromptTemplate`
-   `ChatOpenAI`
-   `StrOutputParser`
-   OpenAI chat model invocation through LangChain
-   Environment-based API key management
-   LangSmith tracing and observability
-   A practical aviation business use case

------------------------------------------------------------------------

## 3. Architecture

``` text
                    Aviation Charter Request
                              |
                              v
                    ChatPromptTemplate
                         (LangChain)
                              |
                              v
                       ChatOpenAI
                         (LangChain)
                              |
                              v
                    StrOutputParser
                         (LangChain)
                              |
                              v
                  Flight Operations Brief
                              |
                              +------------------+
                              |                  |
                              v                  v
                         Terminal           LangSmith
                                           Tracing / UI
```

The core LangChain pipeline uses LCEL:

``` python
chain = prompt | model | parser
```

This creates a composable:

``` text
Prompt → Model → Output Parser
```

pipeline.

------------------------------------------------------------------------

## 4. Technology Stack

  Technology         Purpose
  ------------------ ----------------------------------
  Python             Application language
  LangChain          LLM application framework
  LangChain OpenAI   OpenAI integration for LangChain
  OpenAI             Chat model
  LangSmith          Tracing and observability
  python-dotenv      Loading environment variables

------------------------------------------------------------------------

## 5. Project Structure

``` text
langchain-app/
│
├── main.py
├── .env
├── .gitignore
└── venv/
```

-   **`main.py`** --- Complete application.
-   **`.env`** --- API keys and LangSmith configuration.
-   **`.gitignore`** --- Prevents secrets and the virtual environment
    from being committed.

------------------------------------------------------------------------

## 6. Prerequisites

Make sure you have:

-   Python 3.9+
-   An OpenAI API key
-   A LangSmith API key
-   Internet access

------------------------------------------------------------------------

## 7. Setup

### Step 1: Create the project

``` bash
mkdir langchain-app
cd langchain-app
```

### Step 2: Create a virtual environment

``` bash
python -m venv venv
```

### Step 3: Activate the virtual environment

#### macOS / Linux

``` bash
source venv/bin/activate
```

#### Windows PowerShell

``` powershell
venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

``` cmd
venv\Scripts\activate.bat
```

After activation, the terminal should show `(venv)`.

------------------------------------------------------------------------

## 8. Install Dependencies

``` bash
pip install langchain langchain-openai langsmith python-dotenv
```

------------------------------------------------------------------------

## 9. Configure Environment Variables

Create a `.env` file:

``` env
OPENAI_API_KEY=your-openai-api-key
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=aviation-langchain-assignment
```

### Security

Never hard-code API keys in Python code.

Do not commit `.env` to GitHub.

Add the following to `.gitignore`:

``` gitignore
venv/
.env
__pycache__/
```

------------------------------------------------------------------------

## 10. LangChain Implementation

The application uses LangChain components instead of calling the OpenAI
API directly.

### Prompt Template

``` python
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an aviation charter operations assistant.

        Analyze a customer charter flight request.

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

        {flight_request}
        """
    )
])
```

### Model

``` python
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)
```

### Output Parser

``` python
parser = StrOutputParser()
```

### LCEL Chain

``` python
chain = prompt | model | parser
```

### Invoke the Chain

``` python
result = chain.invoke({
    "flight_request": flight_request
})
```

The application flow is:

``` text
Input
  ↓
Prompt Template
  ↓
OpenAI Chat Model
  ↓
String Output Parser
  ↓
Operations Brief
```

------------------------------------------------------------------------

## 11. Run the Application

Run:

``` bash
python main.py
```

Expected terminal output:

``` text
============================================================
AVIATION CHARTER OPERATIONS ASSISTANT
============================================================

Input:
Customer: ABC Corporation
Route: Chennai → Dubai
Departure: 15 September 2026, 08:30 IST
Passengers: 12

Generating operations brief...

Output:
Flight Operations Brief
...
============================================================
```

The exact model output may vary.

------------------------------------------------------------------------

## 12. LangSmith Tracing

LangSmith is used to observe the LangChain execution.

Tracing is enabled through:

``` env
LANGCHAIN_TRACING_V2=true
```

The project is configured with:

``` env
LANGCHAIN_PROJECT=aviation-langchain-assignment
```

When the application runs, LangSmith captures the LangChain execution.

A trace will show the execution flow, including:

``` text
RunnableSequence
    |
    +-- ChatPromptTemplate
    |
    +-- ChatOpenAI
    |
    +-- StrOutputParser
```

Depending on the integration and available metadata, the trace can
provide:

-   Input
-   Output
-   Model invocation
-   Execution duration
-   Token usage
-   Errors
-   Individual chain steps

------------------------------------------------------------------------

## 13. Why LangSmith Is Useful

Without tracing, the application primarily gives us:

``` text
Input → Output
```

With LangSmith, we gain visibility into the LLM application:

``` text
Input
  ↓
Prompt
  ↓
Model
  ↓
Output Parser
  ↓
Output

       ↘
        LangSmith Trace
```

This becomes particularly valuable as the application grows into a more
complex AI workflow.

------------------------------------------------------------------------

## 14. Business Value

A real aviation operations system may receive requests from customers,
travel coordinators, brokers, or internal teams.

The request may contain:

-   Route
-   Departure and arrival information
-   Passenger count
-   Aircraft requirements
-   Catering requirements
-   Baggage requirements
-   Special services
-   Return flight information
-   Unconfirmed details

An AI assistant can help operations teams quickly identify:

### Confirmed Information

What is already known about the flight.

### Missing Information

What still needs to be collected from the customer.

### Operational Attention

Items that may require coordination with aircraft operators, catering
teams, airport services, or other stakeholders.

This prototype is intentionally small and demonstrates the foundation
for such a workflow.

------------------------------------------------------------------------

## 15. Current Scope

This assignment is intentionally limited to:

``` text
Customer Flight Request
        ↓
LangChain
        ↓
OpenAI
        ↓
Operations Brief
        ↓
LangSmith Trace
```

It does **not** currently:

-   Book flights
-   Query aircraft availability
-   Access a CRM
-   Send emails
-   Query live aviation systems
-   Make operational decisions autonomously
-   Replace human approval

The generated brief is an AI-assisted summary and should be reviewed by
an operations professional before being used operationally.

------------------------------------------------------------------------

## 16. Future Enhancements

This project can be extended into a production-style aviation AI
workflow.

### Phase 1 --- Structured Output

Return a predictable JSON structure:

``` json
{
  "flight_summary": {},
  "passenger_requirements": [],
  "missing_information": [],
  "operational_attention": []
}
```

### Phase 2 --- RAG

Add aviation documents, SOPs, policies, and operational knowledge using
a retrieval-augmented generation pipeline.

``` text
Flight Request
      ↓
Retriever
      ↓
Relevant Aviation Knowledge
      ↓
LLM
      ↓
Operations Brief
```

### Phase 3 --- Tools

Give the application tools to:

-   Search flight information
-   Query internal systems
-   Check aircraft information
-   Retrieve customer information
-   Access operational documents

### Phase 4 --- Agent

Convert the workflow into an agent capable of selecting appropriate
tools based on the request.

### Phase 5 --- Evaluation

Introduce an evaluation dataset and metrics to measure:

-   Correctness
-   Completeness
-   Missing-information detection
-   Hallucination rate
-   Response quality

### Phase 6 --- Production Architecture

``` text
                    Customer / Operations User
                              |
                              v
                         AI Assistant
                              |
                    +---------+---------+
                    |                   |
                    v                   v
                  RAG                 Tools
                    |                   |
                    v                   v
              Knowledge Base      Enterprise APIs
                    \                   /
                     \                 /
                      v               v
                           LLM
                            |
                            v
                     Structured Result
                            |
                            v
                      Human Approval
                            |
                            v
                    Operational System
                            |
                            v
                       LangSmith
                 Tracing + Evaluation
```

------------------------------------------------------------------------

## 17. Key Learning

The main objective is not simply to call an LLM.

The objective is to understand how **LangChain can be used as the
application framework around an LLM**.

The core concepts demonstrated are:

``` text
LangChain
   |
   +-- Prompt Templates
   |
   +-- Model Integration
   |
   +-- LCEL
   |
   +-- Output Parsing
   |
   +-- Invocation
   |
   +-- LangSmith Observability
```

------------------------------------------------------------------------

## 18. Interview Explanation

A concise explanation of the project:

> I built a small aviation charter operations assistant using LangChain.
> The application accepts a customer flight request and uses a LangChain
> pipeline consisting of a prompt template, an OpenAI chat model, and an
> output parser to generate an operations brief. I used environment
> variables for API key management and enabled LangSmith tracing to
> observe the complete LangChain execution, including the input, model
> invocation, output, and execution information. I chose an aviation use
> case because it represents a realistic workflow where customer
> requests contain operational requirements and missing information that
> can be identified and summarized by an LLM.

------------------------------------------------------------------------

## 19. Assignment Requirements Checklist

-   [x] Python application
-   [x] LangChain framework
-   [x] OpenAI chat model
-   [x] OpenAI accessed through LangChain
-   [x] LangChain prompt template
-   [x] LCEL pipeline
-   [x] Output parser
-   [x] Response printed in terminal
-   [x] LangSmith tracing enabled
-   [x] API keys stored in environment variables
-   [x] No hard-coded API keys
-   [x] `.env` excluded from Git
-   [x] Single `main.py` application
-   [x] Real-world aviation use case

------------------------------------------------------------------------

## 20. Conclusion

This project demonstrates a practical first step toward building
**LLM-powered enterprise applications with LangChain**.

Instead of a generic chatbot or trivia example, the application applies
LangChain to a realistic aviation charter workflow and introduces
**observability through LangSmith**.

It provides a foundation that can later be expanded with structured
outputs, RAG, tools, agents, evaluation, and enterprise integrations.
