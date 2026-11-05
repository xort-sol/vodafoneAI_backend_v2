"""Scripted customers for the eval (test_eval.py). Add one for every bad call you hear.

lines: what the customer says, in order (the agent opens the call first).
Optional: lead (lead fields), outcome (expected record_outcome), intent (judged against the last reply),
opening_has (text the first message must contain), callback_tomorrow (callback_time must name tomorrow's date).
"""

INTERESTED_INTERNET = ["Sure, go ahead.", "Our office internet is slow, there's about ten of us.", "Yes, tell me more."]

PERSONAS = [
    # Permission and opening
    {"id": "busy-calls-back", "lines": ["Hello?", "I'm driving, call me tomorrow at 10am."], "outcome": "callback",
     "callback_tomorrow": True},
    {"id": "do-not-call", "lines": ["Who is this?", "Take me off your list and don't call me again."], "outcome": "do_not_call"},
    {"id": "flat-no", "lines": ["Yeah?", "No thanks, not interested.", "No, leave it, thanks."], "outcome": "not_interested"},
    {"id": "hi-only", "lines": ["Hi.", "Hello?"],
     "intent": "Asks whether now is an okay time to talk. Names no service or price."},
    {"id": "is-this-a-robot", "lines": ["Sure.", "Wait, am I talking to a real person?"],
     "intent": "Says honestly that it is an AI assistant, then carries on politely."},
    {"id": "known-name", "lead": {"name": "Mary Byrne", "company": "Byrne Bakery"}, "lines": ["Hello?"], "opening_has": "Mary"},
    {"id": "rude", "lines": ["What do you want?", "Get to the point."],
     "intent": "Stays polite and calm, gives a short reason for the call and asks one question."},
    # Discovery and pitch
    {"id": "slow-internet-yes", "lines": INTERESTED_INTERNET + ["Sounds good.", "Thursday morning around ten, by phone."],
     "outcome": "interested"},
    {"id": "mobile-bill", "lines": ["Go ahead.", "Our mobile bill is way too high for five phones.", "What does it cost?",
                                    "Okay.", "Monday at two works."], "outcome": "interested"},
    {"id": "calls-abroad", "lines": ["Yes.", "We call the UK and the US a lot and it costs a fortune.", "Go on.",
                                     "Yes please, call me Friday at eleven."], "outcome": "interested"},
    {"id": "sites-no-fibre", "lines": ["Okay, quickly.", "We're a building firm, our sites have no internet at all."],
     "intent": "Asks a follow-up or suggests mobile broadband, not fixed fibre broadband."},
    {"id": "happy-customer", "lines": ["Sure.", "Honestly everything works fine.", "No, really, we're happy.", "No thanks."],
     "outcome": "not_interested"},
    {"id": "soft-okay", "lines": ["Yeah. Sure. Go ahead.", "Yeah. That's going good, but not up to the mark here.",
                                  "Maybe low speed. And sometime it it... It's not working.", "Yeah. Obviously.",
                                  "Yeah. Go ahead.", "Okay.", "Wednesday at 4pm."], "outcome": "interested"},
    {"id": "maybe", "lines": INTERESTED_INTERNET + ["Maybe, I'm not sure."],
     "intent": "Asks when would suit a team member to call them. Doesn't pitch more features or say goodbye."},
    {"id": "didnt-catch-that", "lines": ["Sure.", "Our internet keeps dropping.", "Sorry?"],
     "intent": "Repeats its last point in fewer, simpler words and asks the same question again."},
    {"id": "short-answers", "lines": ["Yep.", "Fine.", "Not really."],
     "intent": "Asks what isn't working. Doesn't name a service, plan or price yet."},
    {"id": "frustrated", "lines": ["Yes.", "It's been a nightmare, our internet drops every single afternoon."],
     "intent": "Sounds sympathetic and asks about the problem. Doesn't pitch yet."},
    # Objections (answers in src/agent/objections.md)
    {"id": "too-expensive", "lines": INTERESTED_INTERNET + ["That's too expensive for us."],
     "intent": "Acknowledges the cost and gives one short reframe, such as the ten euro saving or comparing their bill. Doesn't push."},
    {"id": "happy-with-provider", "lines": INTERESTED_INTERNET + ["We're happy with Eir, thanks."],
     "intent": "Accepts that without pressure and suggests comparing when their contract ends, or asks if a follow-up is welcome."},
    {"id": "in-contract", "lines": ["Yes.", "We're in a contract with Three until March, so no point."],
     "intent": "Asks when exactly the contract ends or offers a team member call before it ends. Doesn't pitch harder."},
    {"id": "send-email", "lines": INTERESTED_INTERNET + ["Just send me an email."],
     "intent": "Agrees and asks for the best time or email for a team member to send the details."},
    {"id": "switching-hassle", "lines": INTERESTED_INTERNET + ["Switching sounds like a real hassle."],
     "intent": "Reassures them about switching, for example that most fibre installs happen within fourteen days. "
               "Doesn't promise free installation."},
    # Facts it must not invent
    {"id": "fastest-price", "lines": ["Sure.", "What does your fastest broadband cost?"],
     "intent": "Gives the price of the two gigabit plan, sixty-five euro a month."},
    {"id": "phone-price", "lines": ["Go ahead.", "How much is an iPhone 17 on your plan?"],
     "intent": "Doesn't give a phone price; says a team member will confirm it."},
    {"id": "vat", "lines": ["Yes.", "Does the broadband price include VAT?"],
     "intent": "Doesn't claim either way; says a team member will confirm."},
    {"id": "price-rises", "lines": ["Sure.", "Do your prices go up every year?"],
     "intent": "Honestly says monthly prices rise every April by a small fixed amount."},
    {"id": "big-company", "lines": ["Sure.", "We have about sixty staff across three offices."],
     "intent": "Offers a team member callback about a custom plan."},
    {"id": "off-topic", "lines": ["Sure.", "Can you help me with my electricity bill?"],
     "intent": "Politely says it can't help with that and steers back to phones or internet."},
    # Lead context
    {"id": "contract-ending", "lead": {"name": "Tom Walsh", "company": "Walsh Dental", "staff": "12",
                                       "current_plan": "Eir fibre 300", "contract_end": "December 2026"},
     "lines": ["Yes, go ahead."],
     "intent": "Asks a question that builds on their current internet or their contract ending soon."},
    {"id": "big-team-mobile", "lead": {"staff": "40", "notes": "asked about phones for the sales team"},
     "lines": ["Yeah, I've a minute."],
     "intent": "Asks about their team's phones or mobile needs."},
]
