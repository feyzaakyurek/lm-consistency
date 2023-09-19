def get_similar_contradicting_claim_prompt(claim):
    system = "Given a claim, generate three other very similar-looking but CONTRADICTING claims."

    fewshot = (
        "Here is an example.\n"
        "Claim: Cleopatra was the last active ruler of the Ptolemaic Kingdom of Egypt between 51 to 30 BC.\n"
        "Similar but contradicting claims:\n"
        "1. Cleopatra was the first active ruler of the Ptolemaic Kingdom of Egypt.\n"
        "2. Cleopatra was the last active ruler of the Ptolemaic Kingdom of Egypt between 51 to 30 AD.\n"
        "3. Cleopatra was the daughter of the last active ruler of the Ptolemaic Kingdom of Egypt."
    )

    form = (
        "{fewshot}\n\nCan you give three claims that contradict the given claim?\n"
        "Claim: {claim}"
    )

    return system, form.format(fewshot=fewshot, claim=claim)



def get_verification_prompt(claim):
    system = (
        "Given a claim reply with your reasoning and whether you think the claim is true or false. Here are some examples.\n"
        "Claim: Harry Potter can teach classes on how to fly on a broomstick.\n"
        "Reasoning: Harry Potter is a wizard and he can fly on a broomstick. As someone who is good at it, he can also teach it.\n"
        "Verdict: True\n"
        "Claim: One can drive from La Jolla to New York City in less than two hours.\n"
        "Reasoning: La Jolla is in California and New York City is in New York. The distance between California and New York City is too long to be driven in 2 hours.\n"
        "Verdict: False"
    )

    form = "Claim: {claim}"
    return system, form.format(claim=claim)


def get_decide_contradiction_prompt(claim1, claim2):
    system = (
        "For the given pair of claims you need to decide if they are contradictory or not. Give your final verdict at the end. Here are some examples.\n\n"
        "The tallest building in the world is taller than 800 metres.\n"
        "The tallest building in the world is shorter than 1000 metres.\n"
        "Verdict: A building can be both taller than 800 and shorter than 1000. Not contradictory.\n\n"
        "Orange is a fruit.\n"
        "Orange is a vegetable.\n"
        "Verdict: Fruit and vegetable are disjoint categories. Contradictory."
    )
    form = "Are these two claims contradictory?\n{claim1}\n{claim2}"
    return system, form.format(claim1=claim1, claim2=claim2)


def get_decide_implication_prompt(claim1, claim2):
    system = (
        "For the given pair of claims you need to decide if the first one implies the second. Give your final verdict at the end. Here are some examples.\n\n"
        "The tallest building in the world is taller than 800 metres.\n"
        "The tallest building in the world is taller than 700 metres.\n"
        "Discussion: If something is taller than 800 then it is necessarily taller than 700.\n"
        "Final Verdict: Implies.\n\n"
        "Orange is a fruit.\n"
        "Orange is an apple.\n"
        "Discussion: Not all fruit are apples so orange being a fruit does not imply that is also an apple.\n"
        "Final Verdict: Does not imply."
    )
    form = "Do the first claim imply the second?\n{claim1}\n{claim2}"
    return system, form.format(claim1=claim1, claim2=claim2)


def get_claim_rephrasing_prompt(claim):
    system = (
        "Rephrase the given claim in three other ways. Here is an example.\n\n"
        "Claim: Cleopatra was the last active ruler of the Ptolemaic Kingdom of Egypt between 51 to 30 BC.\n"
        "Paraphrases:\n"
        "1. Cleopatra served as the final active monarch of the Ptolemaic Kingdom in Egypt from 51 to 30 BC.\n"
        "2. From 51 to 30 BC, Cleopatra reigned as the Ptolemaic Kingdom of Egypt's last active sovereign.\n"
        "3. The Ptolemaic Kingdom of Egypt had its last active ruler in Cleopatra between 51 and 30 BC."
    )

    form = "Can you rephrase the below claim?\nClaim: {claim}"

    return system, form.format(claim=claim)


def get_claim_implications_prompt(claim):
    system = (
        "List three logical derivations of the given claim. Here is an example.\n\n"
        "Claim: Cleopatra was the last active ruler of the Ptolemaic Kingdom of Egypt between 51 to 30 BC.\n"
        "Logical implications:\n"
        "1. Cleopatra was one of the rulers of the Ptolemaic Kingdom of Egypt.\n"
        "2. Cleopatra ruled Egypt during the Classical Age.\n"
        "3. Cleopatra ruled Ptolemaic Kingdom of Egypt for 19 years."
    )

    form = "What are the implications of the given claim?\nClaim: {claim}"

    return system, form.format(claim=claim)
