"""One-off helper that writes ``eval/golden_set.jsonl``.

Kept in the repo so the golden set is **reproducible and reviewable** rather than
a file of unexplained labels. Every ``expected_answer`` below is taken from the
corpus text itself, not from outside knowledge of these schemes — the eval must
measure whether *this* corpus is retrieved and used faithfully, not whether the
model happens to know Indian agriculture.

Every ``relevant_chunk_ids`` entry is verified to exist in
``data/processed/documents.jsonl`` before anything is written; a label pointing
at a chunk that no longer exists silently scores zero and looks like a
retrieval failure.

Re-run after any change to chunking:

    python eval/build_golden_set.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DOCUMENTS_JSONL = PROJECT_ROOT / "data" / "processed" / "documents.jsonl"
GOLDEN_SET_PATH = PROJECT_ROOT / "eval" / "golden_set.jsonl"

GOLDEN: list[dict] = [
    # ── eligibility ──────────────────────────────────────────────────────────
    dict(
        question="Who is eligible for the Kisan Credit Card?",
        expected_answer=(
            "All agriculture clients who have a good track record for the last "
            "two years are eligible for the Kisan Credit Card Scheme (KCCS)."
        ),
        relevant_chunk_ids=["tnau-paddy-schemes::0060"],
        scheme="Kisan Credit Card",
        question_type="eligibility",
    ),
    dict(
        question="Who can get help under the Land Development Scheme?",
        expected_answer="Farmers of all categories are eligible under the Land Development Scheme.",
        relevant_chunk_ids=["tnau-paddy-schemes::0021"],
        scheme="Land Development Scheme",
        question_type="eligibility",
    ),
    dict(
        question="Which farmers can register their seed farms under the seed production programme?",
        expected_answer=(
            "All farmers who produce and supply seeds to the Department of Agriculture "
            "on a contract basis can enrol and register their seed farms. Preference is "
            "given to farm women groups and Farmers Interest Groups."
        ),
        relevant_chunk_ids=["tnau-paddy-schemes::0001", "tnau-paddy-schemes::0002"],
        scheme="Seed Multiplication Scheme",
        question_type="eligibility",
    ),
    dict(
        question="Who is eligible for the quality seed distribution subsidy?",
        expected_answer=(
            "All farmers are eligible, with preference given to small and marginal women "
            "farmers, and 30% of the benefit assured to SC/ST farmers."
        ),
        relevant_chunk_ids=["tnau-paddy-schemes::0004"],
        scheme="Seed Village Programme",
        question_type="eligibility",
    ),
    dict(
        question="Is the National Agricultural Insurance Scheme funded by the centre or the state?",
        expected_answer=(
            "It is sponsored by both the Central and State Government, on a 50:50 "
            "sharing basis."
        ),
        relevant_chunk_ids=["tnau-paddy-schemes::0035"],
        scheme="National Agricultural Insurance Scheme",
        question_type="eligibility",
    ),
    # ── subsidy_amount ───────────────────────────────────────────────────────
    dict(
        question="What is the minimum credit limit under the Kisan Credit Card?",
        expected_answer=(
            "The minimum credit limit is Rs. 3000. The credit limit is based on "
            "operational land holding, cropping pattern and scale of finance."
        ),
        relevant_chunk_ids=["tnau-paddy-schemes::0061"],
        scheme="Kisan Credit Card",
        question_type="subsidy_amount",
    ),
    dict(
        question="What subsidy is given for replacing an old pumpset below 5 HP?",
        expected_answer=(
            "For pump sets below 5 HP, SC/ST farmers get a subsidy of Rs.3500 or 50% of "
            "the cost, whichever is less; other farmers get Rs.2500 or 25% of the cost, "
            "whichever is less."
        ),
        relevant_chunk_ids=["tnau-paddy-schemes::0028", "tnau-paddy-schemes::0029"],
        scheme="Replacement of old Pumpsets",
        question_type="subsidy_amount",
    ),
    dict(
        question="How much subsidy is given for a pumpset of 5 HP and above?",
        expected_answer=(
            "For pump sets of 5 HP and above, SC/ST farmers get Rs.6000 or 50% of the "
            "cost, whichever is less; other farmers get Rs.5000 or 25% of the cost, "
            "whichever is less."
        ),
        relevant_chunk_ids=["tnau-paddy-schemes::0029"],
        scheme="Replacement of old Pumpsets",
        question_type="subsidy_amount",
    ),
    dict(
        question="How much money does PM-KISAN give a farmer each year?",
        expected_answer=(
            "PM-KISAN provides an annual payment of Rs 6,000 in three equal instalments, "
            "deposited straight into the bank account."
        ),
        relevant_chunk_ids=["testbook-agri-schemes::0042"],
        scheme="PM-KISAN",
        question_type="subsidy_amount",
    ),
    dict(
        question="What is the farmer's premium share under Pradhan Mantri Fasal Bima Yojana?",
        expected_answer=(
            "The farmer pays a low premium of 2% for Kharif crops and 1.5% for Rabi "
            "crops. The balance premium is paid by the Central and State governments."
        ),
        relevant_chunk_ids=["testbook-agri-schemes::0019"],
        scheme="Pradhan Mantri Fasal Bima Yojana",
        question_type="subsidy_amount",
    ),
    dict(
        question="What subsidy is available on paddy seeds through the seed village concept?",
        expected_answer=(
            "A subsidy of Rs.5 per kg of paddy seeds is allowed in the sale price at "
            "Agricultural Extension Centres, or 50% of cost, whichever is less."
        ),
        relevant_chunk_ids=["tnau-paddy-schemes::0004"],
        scheme="Seed Village Programme",
        question_type="subsidy_amount",
    ),
    dict(
        question="What incentive is paid for producing certified class paddy seed?",
        expected_answer="A premium of Rs.2 per kg of seed is paid for the production of certified class seeds.",
        relevant_chunk_ids=["tnau-paddy-schemes::0002"],
        scheme="Seed Multiplication Scheme",
        question_type="subsidy_amount",
    ),
    dict(
        question="How much subsidy is given for a System of Rice Intensification demonstration?",
        expected_answer=(
            "A subsidy of Rs 3000 per demonstration of 0.4 hectare is given, covering "
            "improved seed, conoweeder, marker, bio fertilizers and micro nutrient mixture."
        ),
        relevant_chunk_ids=["tnau-paddy-schemes::0006"],
        scheme="National Food Security Mission",
        question_type="subsidy_amount",
    ),
    dict(
        question="What subsidy is available for buying agricultural machinery in Tamil Nadu?",
        expected_answer=(
            "Under the Agricultural Mechanisation Programme, 25% of the cost of the "
            "machinery or implement is given, or the ceiling limit prescribed by the "
            "Government of India, whichever is less."
        ),
        relevant_chunk_ids=["tnau-paddy-schemes::0019"],
        scheme="Agricultural Mechanisation Programme",
        question_type="subsidy_amount",
    ),
    dict(
        question="How much subsidy is given for a storage bin?",
        expected_answer=(
            "Rs.3000 for a 20 quintal bin and Rs.1500 for a 10 quintal bin, or 33% of "
            "the cost of the bin."
        ),
        relevant_chunk_ids=["tnau-paddy-schemes::0012", "tnau-paddy-schemes::0013"],
        scheme="Seed Village Programme",
        question_type="subsidy_amount",
    ),
    dict(
        question="What grant is given for rain water harvesting works on community land?",
        expected_answer=(
            "A 100% grant is provided on the cost of works taken up in community lands, "
            "though beneficiaries contribute 10% of the cost (5% for SC/ST). On patta "
            "lands 90% subsidy is provided."
        ),
        relevant_chunk_ids=["tnau-paddy-schemes::0026"],
        scheme="Rain Water Harvesting Programme",
        question_type="subsidy_amount",
    ),
    dict(
        question="How much can a farmer borrow under the Pledge Loan Scheme?",
        expected_answer=(
            "Loans up to Rs. 50,000 or 60% of the value of the agricultural produce may "
            "be paid against the pledge of produce. No interest is charged for the first "
            "30 days."
        ),
        relevant_chunk_ids=["tnau-paddy-schemes::0057"],
        scheme="Pledge Loan Scheme",
        question_type="subsidy_amount",
    ),
    dict(
        question="What is the provision for a Farmers Field School training?",
        expected_answer=(
            "A lump sum provision of Rs.17,000 is made, which includes honorarium, "
            "training material and conduct of field days."
        ),
        relevant_chunk_ids=["tnau-paddy-schemes::0008"],
        scheme="Integrated Pest Management",
        question_type="subsidy_amount",
    ),
    dict(
        question="What is the outlay of the Kerala crop insurance scheme?",
        expected_answer=(
            "The outlay of the Kerala Crop Insurance Scheme is Rs.100.00 lakhs. It is "
            "sponsored and 100% funded by the State Government."
        ),
        relevant_chunk_ids=["tnau-paddy-schemes::0033"],
        scheme="Crop Insurance Scheme",
        question_type="subsidy_amount",
    ),
    # ── deadline / validity ──────────────────────────────────────────────────
    dict(
        question="How long is a Kisan Credit Card valid?",
        expected_answer="The Kisan Credit Card is valid for 3 years, subject to annual review.",
        relevant_chunk_ids=["tnau-paddy-schemes::0061"],
        scheme="Kisan Credit Card",
        question_type="deadline",
    ),
    dict(
        question="How long is the interest-free period on a pledge loan?",
        expected_answer=(
            "No interest is charged for the first 30 days. Interest at 8% and 12.5% is "
            "charged for the next two successive months, and the scheme runs for a short "
            "period of 90 days."
        ),
        relevant_chunk_ids=["tnau-paddy-schemes::0057"],
        scheme="Pledge Loan Scheme",
        question_type="deadline",
    ),
    dict(
        question="How quickly is machinery allotted under the Land Development Scheme?",
        expected_answer="Machinery is allotted on a priority basis under the Land Development Scheme.",
        relevant_chunk_ids=["tnau-paddy-schemes::0021"],
        scheme="Land Development Scheme",
        question_type="deadline",
    ),
    # ── documents_required / where to apply ──────────────────────────────────
    dict(
        question="What personal insurance cover comes with a Kisan Credit Card?",
        expected_answer=(
            "The Kisan Credit Card covers personal insurance against death or permanent "
            "disability for a maximum of Rs. 50,000 and Rs. 25,000 respectively."
        ),
        relevant_chunk_ids=["tnau-paddy-schemes::0061"],
        scheme="Kisan Credit Card",
        question_type="documents_required",
    ),
    dict(
        question="Which officer should a farmer contact for the seed multiplication scheme?",
        expected_answer=(
            "The Assistant Agricultural Officer at the Village Level; at block level the "
            "Assistant Seed Officer, Deputy Agricultural Officer or Agricultural Officer."
        ),
        relevant_chunk_ids=["tnau-paddy-schemes::0001", "tnau-paddy-schemes::0002"],
        scheme="Seed Multiplication Scheme",
        question_type="documents_required",
    ),
    dict(
        question="Where should a farmer apply for the agricultural mechanisation subsidy?",
        expected_answer=(
            "Farmers should approach the Assistant Executive Engineer, AED in the Revenue "
            "Division; the Executive Engineer, AED in the District; the Superintending "
            "Engineer, AED in the Region; or the Chief Engineer, Agricultural Engineering "
            "Department, Nandanam, Chennai-35."
        ),
        relevant_chunk_ids=["tnau-paddy-schemes::0019"],
        scheme="Agricultural Mechanisation Programme",
        question_type="documents_required",
    ),
]


def main() -> None:
    """Verify every labelled chunk id, then write the golden set."""
    if not DOCUMENTS_JSONL.exists():
        raise SystemExit(
            f"{DOCUMENTS_JSONL} not found. Build the corpus first: "
            "python ingestion/build_index.py"
        )

    known = {
        json.loads(line)["metadata"]["chunk_id"]
        for line in DOCUMENTS_JSONL.open(encoding="utf-8")
        if line.strip()
    }
    missing = sorted(
        {cid for item in GOLDEN for cid in item["relevant_chunk_ids"] if cid not in known}
    )
    if missing:
        raise SystemExit(
            "Refusing to write: these labelled chunk_ids are not in the current "
            f"corpus (chunking probably changed): {missing}"
        )

    with GOLDEN_SET_PATH.open("w", encoding="utf-8") as handle:
        for item in GOLDEN:
            handle.write(json.dumps({**item, "language": "en"}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(GOLDEN)} questions to {GOLDEN_SET_PATH}")
    print("  by question_type:", dict(collections.Counter(i["question_type"] for i in GOLDEN)))
    print("  distinct schemes:", len({i["scheme"] for i in GOLDEN}))
    print("  chunk ids verified:", sum(len(i["relevant_chunk_ids"]) for i in GOLDEN))


if __name__ == "__main__":
    main()
