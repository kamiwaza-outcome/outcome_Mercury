"""
Standard questions for RFP refinement based on common gaps in generated documents
"""

STANDARD_RFP_QUESTIONS = [
    {
        "category": "Company Information",
        "questions": [
            "What is your company's full legal name and CAGE code?",
            "What is your company's UEI (Unique Entity Identifier)?",
            "What is your primary point of contact's name, email, and phone number?",
            "What is your company's physical address and remit-to address?"
        ]
    },
    {
        "category": "Compliance & Certifications",
        "questions": [
            "Are you an approved source or authorized distributor for this item? If yes, please provide details.",
            "Do you have ISO 9001, AS9100, or other quality certifications? Please list them.",
            "Can you provide Certificate of Conformance (CoC) with shipments?",
            "Are you registered in SAM.gov? What is your registration status?"
        ]
    },
    {
        "category": "Technical Capabilities",
        "questions": [
            "What is your typical lead time for this item?",
            "What is your minimum order quantity (MOQ)?",
            "Can you meet the full quantity requirement? If not, what quantity can you provide?",
            "Do you maintain inventory for this item? If yes, current stock levels?"
        ]
    },
    {
        "category": "Pricing & Delivery",
        "questions": [
            "What is your unit price for the requested quantity?",
            "Do you offer quantity discounts? If yes, please provide pricing tiers.",
            "What are your standard payment terms (Net 30, Net 60, etc.)?",
            "Can you deliver to multiple locations if required? Any additional costs?"
        ]
    },
    {
        "category": "Past Performance",
        "questions": [
            "Please provide 2-3 relevant past contracts with government agencies (contract numbers, POCs).",
            "Have you previously supplied this or similar items to DoD/Navy? Please provide details.",
            "What is your typical on-time delivery rate percentage?",
            "Have you had any quality issues or contract terminations in the past 3 years?"
        ]
    },
    {
        "category": "Additional Capabilities",
        "questions": [
            "Do you offer any value-added services (kitting, custom packaging, etc.)?",
            "Can you support urgent/expedited orders? What is the additional cost?",
            "Do you have experience with WAWF (Wide Area WorkFlow) for invoicing?",
            "Are there any special considerations or advantages your company offers for this procurement?"
        ]
    }
]

def get_all_questions_flat():
    """Return all questions as a flat list with category info"""
    questions = []
    for category_group in STANDARD_RFP_QUESTIONS:
        for question in category_group["questions"]:
            questions.append({
                "question": question,
                "category": category_group["category"],
                "context": "",
                "type": "standard"
            })
    return questions

def get_essential_questions():
    """Return only the most essential questions for quick review"""
    essential = [
        {
            "question": "What is your company's full legal name and CAGE code?",
            "category": "Company Information",
            "context": "",
            "type": "essential"
        },
        {
            "question": "Are you an approved source or authorized distributor for this item? If yes, please provide details.",
            "category": "Compliance",
            "context": "",
            "type": "essential"
        },
        {
            "question": "What is your unit price for the requested quantity?",
            "category": "Pricing",
            "context": "",
            "type": "essential"
        },
        {
            "question": "What is your typical lead time for this item?",
            "category": "Delivery",
            "context": "",
            "type": "essential"
        },
        {
            "question": "Please provide 2-3 relevant past contracts with government agencies (contract numbers, POCs).",
            "category": "Past Performance",
            "context": "",
            "type": "essential"
        }
    ]
    return essential