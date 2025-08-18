Team name :  INNOVATORS




<img width="516" height="141" alt="image" src="https://github.com/user-attachments/assets/02411381-d312-493e-8e6e-c7de5c8acc2f" />


Project Description – Bargain AI: Intelligent Negotiation Between Buyer & Seller Agents

This project implements a negotiation system powered by AI agents, designed to simulate realistic buyer–seller bargaining scenarios. The goal is to model how automated agents can negotiate prices, exchange offers, and make decisions under budget and market constraints.

🔹 Key Components

1. Buyer Agent (YourBuyerAgent)

Represents a pragmatic, analytical buyer.

Personality traits: data-driven, calm, firm, and time-sensitive.

Always respects its budget limit and never exceeds it.

Starts with evidence-based low offers and gradually concedes using a concession schedule as rounds progress.

Evaluates offers based on market price, product quality (A, B, Export), quantity, and perishability.

Decision strategies include accepting fair offers, making counteroffers, or rejecting unrealistic prices.



2. Seller Agent (Mock Seller)

Simulates a seller who starts with a high opening price (typically 1.5× market price).

Concedes step by step depending on buyer responses, but never drops below a defined minimum acceptable price.

Can give final offers in later rounds, mimicking real-world negotiation tactics.



3. Negotiation Context

Captures details of the product (type, quality, origin, base price, attributes).

Maintains history of offers, messages, and negotiation rounds.

Enforces a 10-round negotiation limit to simulate time-bound decision making.



4. Internal Components

MemoryComponent: Tracks conversation history, last offers, and computes statistics (mean seller/buyer offers).

ObservationComponent: Extracts prices from textual messages.

DecisionComponent: Determines fair prices, calculates counteroffers, and ensures budget compliance.




🔹 How It Works

1. Seller opens with a high price.


2. Buyer generates an opening offer based on market analysis and budget.


3. Rounds of bargaining continue:

If the seller’s price ≤ buyer’s fair price threshold, the buyer accepts.

Otherwise, the buyer computes a counteroffer based on concession rules.

If nearing timeout (10 rounds), buyer may accept slightly higher offers to avoid losing the deal.

If seller’s demand is beyond budget, buyer issues a firm limit or walks away.



4. The negotiation ends with ACCEPTED, REJECTED, or TIMEOUT status.



🔹 Features

Realistic price bargaining with personality-driven behavior.

Dynamic concession schedules and budget enforcement.

Handles different product qualities, perishable goods, and bulk discounts.

Produces a full conversation transcript of offers and counteroffers.

Can be extended to integrate with multi-agent frameworks (e.g., Concordia).



Output - buyer agent:



![WhatsApp Image 2025-08-17 at 20 08 16_d8801126](https://github.com/user-attachments/assets/36758127-8764-4809-9d44-987bad3f7c86)


![WhatsApp Image 2025-08-17 at 20 12 37_9a889ba8](https://github.com/user-attachments/assets/1ce60096-a393-4e7a-bfe3-65732df55813)




Output - Seller agent:



![WhatsApp Image 2025-08-18 at 22 30 46_c5424f45](https://github.com/user-attachments/assets/7c3184c4-3b4c-4103-98db-0344a8e84692)


![WhatsApp Image 2025-08-18 at 22 31 10_5acbffdc](https://github.com/user-attachments/assets/43721f07-5a57-40c9-9151-00e82b0cd40a)




