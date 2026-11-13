"""Scripted customers for the eval (test_eval.py). Add one for every bad call you hear.

lines: what the customer says, in order (the agent opens the call first).
Optional: lead (lead fields), outcome (expected record_outcome), intent (judged against the last reply),
opening_has (text the first message must contain), callback_tomorrow (callback_time must name tomorrow's date).
"""

# A customer who has answered the discovery questions and confirmed the summary
DISCOVERED = ["Sure, go ahead.", "We're an accountants, six staff, one office, and I make the decisions.",
              "Eir broadband, about sixty a month, and it drops most afternoons.", "Yes, that's right."]

PERSONAS = [
    # Permission and opening
    {"id": "busy-calls-back", "lines": ["Hello?", "I'm driving, call me tomorrow at 10am."], "outcome": "callback",
     "callback_tomorrow": True},
    {"id": "do-not-call", "lines": ["Who is this?", "Take me off your list and don't call me again."], "outcome": "do_not_call"},
    {"id": "hi-only", "lines": ["Hi.", "Hello?"],
     "intent": "Asks whether they've got a few minutes. Names no service or price."},
    {"id": "whats-this-about", "lines": ["What's this about?"],
     "intent": "Says it's a free review of their broadband and mobiles, then asks if they've a few minutes. Names no plan or price."},
    {"id": "is-this-a-robot", "lines": ["Sure.", "Wait, am I talking to a real person?"],
     "intent": "Says honestly that it is an AI assistant, then carries on politely."},
    {"id": "known-name", "lead": {"name": "Mary Byrne", "company": "Byrne Bakery"}, "lines": ["Hello?"], "opening_has": "Mary"},
    {"id": "rude", "lines": ["What do you want?", "Get to the point."],
     "intent": "Stays polite and calm, gives a short reason for the call and asks one question."},
    # Discovery
    {"id": "short-answers", "lines": ["Yep.", "Fine.", "Not really."],
     "intent": "Asks what isn't working. Doesn't name a service, plan or price."},
    {"id": "frustrated", "lines": ["Yes.", "It's been a nightmare, our internet drops every single afternoon."],
     "intent": "Sounds sympathetic and asks about the problem. Doesn't sell anything."},
    {"id": "didnt-catch-that", "lines": ["Sure.", "Our internet keeps dropping.", "Sorry?"],
     "intent": "Repeats its last point in fewer, simpler words and asks the same question again."},
    {"id": "summary-before-next-step", "lines": DISCOVERED[:3] + ["Yes, I'd like someone to look at it."],
     "intent": "Sums up what it learned and asks whether it has got that right, before suggesting any next step."},
    # Next steps
    {"id": "callback-agreed", "lines": DISCOVERED + ["Yes, a call would be good.", "Tomorrow.", "Morning, around ten."],
     "outcome": "interested", "callback_tomorrow": True},
    {"id": "wants-transfer-now", "lines": DISCOVERED + ["Can I talk to someone about it right now?"],
     "intent": "Offers to put them through to a specialist now, or, if it can't, offers a call from a team member soon. "
               "Doesn't say goodbye."},
    {"id": "field-visit-candidate",
     "lines": ["Sure.", "We've three offices in Cork and Limerick, broadband in each and about twenty mobiles.",
               "Eir and Three, it's a mess honestly, I decide.", "Yes, that's right.", "I'd rather meet someone in person."],
     "intent": "Offers a visit from a team member and asks when suits, the address or who they'll meet. "
               "Doesn't say the visit is booked or confirmed."},
    {"id": "near-renewal", "lead": {"company": "Walsh Dental", "current_plan": "Eir fibre 300"},
     "lines": ["Yes, go ahead.", "Our Eir contract ends in January, we'd compare before then."],
     "intent": "Asks when exactly the contract ends or offers a comparison before it renews. Doesn't pitch a plan."},
    # Objections (answers in src/agent/objections.md)
    {"id": "happy-with-provider", "lines": ["Sure.", "We're happy with Eir, thanks."],
     "intent": "Accepts that without pressure or criticising Eir, and asks when their contract is up."},
    {"id": "already-vodafone", "lines": ["Sure.", "We're already with Vodafone."],
     "intent": "Thanks them and asks what they have now or whether it still fits how they work. No pressure."},
    {"id": "send-email", "lines": ["Sure.", "Just send me an email."],
     "intent": "Agrees and asks which services they'd like details on, or their business email. Doesn't say anything was sent."},
    {"id": "no-need", "lines": ["Yeah?", "We don't need anything.", "No, honestly, we're fine."], "outcome": "not_interested"},
    # Facts it must not invent
    {"id": "asks-prices", "lines": ["Sure.", "What does your broadband cost?"],
     "intent": "Gives an exact broadband price including what it rises to after the first twelve months."},
    {"id": "phone-price", "lines": ["Go ahead.", "How much is an iPhone 17 on your plan?"],
     "intent": "Doesn't give a phone price; says a team member will confirm it."},
    {"id": "vat", "lines": ["Yes.", "Does the broadband price include VAT?"],
     "intent": "Doesn't claim either way; says a team member will confirm."},
    {"id": "m365-price", "lines": ["Sure.", "How much is Microsoft 365 with you?"],
     "intent": "Gives no price; says a specialist will go through it."},
    {"id": "price-rises", "lines": ["Sure.", "Do your prices go up every year?"],
     "intent": "Honestly says monthly prices rise every April by a small fixed amount."},
    {"id": "off-topic", "lines": ["Sure.", "Can you help me with my electricity bill?"],
     "intent": "Politely says it can't help with that and steers back to broadband or mobiles."},
    # From call 22 (Orla, a physio on three sites): no transfer possible here (no phone line), and options she turned down
    {"id": "orla-wants-transfer-now",
     "lines": ["Yes.", "We're a physio, five people, three locations, and it's me who decides.",
               "Broadband's slow at one of the sites. Mobiles are with Eir, five of them.", "It's on January twenty fifth.",
               "Can you put me through to someone now?"],
     "intent": "Says in plain words why it can't put them through right now and offers a callback with a team member. "
               "Doesn't mention settings or configuration, and doesn't say goodbye."},
    {"id": "rejects-callback-options", "lines": DISCOVERED + ["Yeah, sure, a call is fine.", "No.", "No, neither.", "Monday at 5pm."],
     "intent": "Confirms a call on Monday at five pm, or says a team member will ring then. "
               "Doesn't ask whether morning or afternoon suits."},
    # Lead context
    {"id": "known-facts-skipped", "lead": {"name": "Tom Walsh", "company": "Walsh Dental", "staff": "12",
                                           "current_plan": "Eir fibre 300"},
     "lines": ["Yes, go ahead."],
     "intent": "Asks a question that doesn't ask for the staff count or who their broadband provider is."},
]
