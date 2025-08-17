# negotiation_agent_ARUNI_ollama.py
import re
import statistics
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ollama_lm import OllamaLanguageModel


@dataclass
class Product:
    name: str
    category: str
    quantity: int
    quality_grade: str
    origin: str
    base_market_price: int
    attributes: Dict[str, Any]


@dataclass
class NegotiationContext:
    product: Product
    your_budget: int
    current_round: int
    seller_offers: List[int]
    your_offers: List[int]
    messages: List[Dict[str, str]]


class DealStatus(Enum):
    ONGOING = "ongoing"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


class BaseBuyerAgent:
    def __init__(self, name: str):
        self.name = name
        self.personality = self.define_personality()

    def define_personality(self) -> Dict[str, Any]:
        raise NotImplementedError

    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        raise NotImplementedError

    def respond_to_seller_offer(self, context: NegotiationContext, seller_price: int, seller_message: str) -> Tuple[DealStatus, int, str]:
        raise NotImplementedError

    def get_personality_prompt(self) -> str:
        raise NotImplementedError


class MemoryComponent:
    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def add(self, role: str, price: Optional[int], message: str, round_no: Optional[int] = None):
        self.history.append({
            "role": role,
            "price": price,
            "message": message,
            "round": round_no
        })

    def last_seller_offer(self) -> Optional[int]:
        for entry in reversed(self.history):
            if entry["role"] == "seller" and entry["price"] is not None:
                return entry["price"]
        return None

    def last_buyer_offer(self) -> Optional[int]:
        for entry in reversed(self.history):
            if entry["role"] == "buyer" and entry["price"] is not None:
                return entry["price"]
        return None

    def rounds(self) -> int:
        rounds = set()
        for e in self.history:
            if e.get("round") is not None:
                rounds.add(e.get("round"))
        return max(rounds) if rounds else 0

    def summary(self) -> Dict[str, Any]:
        seller_prices = [h["price"] for h in self.history if h["role"] == "seller" and h["price"] is not None]
        buyer_prices = [h["price"] for h in self.history if h["role"] == "buyer" and h["price"] is not None]
        return {
            "seller_mean": statistics.mean(seller_prices) if seller_prices else None,
            "buyer_mean": statistics.mean(buyer_prices) if buyer_prices else None,
            "seller_history": seller_prices,
            "buyer_history": buyer_prices,
        }


class ObservationComponent:
    @staticmethod
    def parse_price_from_text(text: str) -> Optional[int]:
        if not text:
            return None
        m = re.search(r'(?:₹|INR|Rs\.?)\s*([0-9,]+)', text, flags=re.IGNORECASE)
        if not m:
            m = re.search(r'([0-9]{4,})', text)
        if m:
            num_str = m.group(1).replace(",", "")
            try:
                return int(num_str)
            except ValueError:
                return None
        return None


class DecisionComponent:
    def __init__(self, personality: Dict[str, Any]):
        self.personality = personality

    def compute_fair_price(self, product: Product) -> int:
        base = product.base_market_price
        grade_factor = {"A": 0.95, "B": 0.85, "Export": 1.05}
        perishability_penalty = 0.98 if product.category.lower() in ["mangoes", "fruits", "perishables"] else 1.0
        quantity_discount = 0.995 if product.quantity >= 100 else 1.0
        fair = int(base * grade_factor.get(product.quality_grade, 0.9) * perishability_penalty * quantity_discount)
        return fair

    def propose_counter(self, context: NegotiationContext, memory: MemoryComponent) -> int:
        seller_last = memory.last_seller_offer() or context.product.base_market_price
        buyer_last = memory.last_buyer_offer() or int(context.product.base_market_price * 0.7)
        round_no = context.current_round
        max_rounds = 10
        concession_factor = min(0.15 + (round_no / max_rounds) * 0.45, 0.6)
        target = int(buyer_last + concession_factor * (seller_last - buyer_last))
        target = min(target, context.your_budget)
        if round_no < max_rounds - 2 and target > seller_last:
            target = min(seller_last - 1000, context.your_budget)
        floor = int(context.product.base_market_price * 0.5)
        if target < floor:
            target = floor
        return target


class YourBuyerAgent(BaseBuyerAgent):
    def __init__(self, name: str, use_llm_persona: bool = False):
        super().__init__(name)
        self.memory = MemoryComponent()
        self.observer = ObservationComponent()
        self.decision = DecisionComponent(self.personality)
        self.use_llm_persona = use_llm_persona
        self.lm = OllamaLanguageModel(model_name="llama3:8b") if use_llm_persona else None

    def define_personality(self) -> Dict[str, Any]:
        return {
            "personality_type": "pragmatic-analytical",
            "traits": ["data-driven", "calm", "time-sensitive", "firm"],
            "negotiation_style": "Starts with evidence-based low offers, concedes predictably as rounds progress, and closes promptly when the deal meets fair value or budget constraints.",
            "catchphrases": ["Let's keep this efficient.", "I value clarity and quick closes."]
        }

    def _persona_message(self, system_ctx: str, seller_msg: str, content: str) -> str:
        """
        If LLM is enabled, rewrite `content` in persona style; else return content.
        """
        if not self.use_llm_persona or self.lm is None:
            return content
        prompt = f"""System:
{system_ctx}

Seller said: {seller_msg}

Instruction:
Rewrite the buyer's message in the given persona, keep it concise, and do not change any numbers.

Original message:
{content}
"""
        try:
            return self.lm.generate(prompt, stop=None, temperature=0.4)
        except Exception:
            # Fallback to original if Ollama is unavailable
            return content

    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        fair = self.decision.compute_fair_price(context.product)
        grade_opening_discount = {"A": 0.30, "B": 0.35, "Export": 0.25}
        discount = grade_opening_discount.get(context.product.quality_grade, 0.32)
        opening_price = int(fair * (1 - discount))
        opening_price = min(opening_price, context.your_budget)
        min_offer = int(context.product.base_market_price * 0.4)
        if opening_price < min_offer:
            opening_price = min_offer
        base_msg = (
            f"{self.personality['catchphrases'][0]} I can start at ₹{opening_price:,} for "
            f"{context.product.quantity} x {context.product.name}. "
            f"I'm data-driven and prefer a quick close if the numbers make sense."
        )
        # No seller message yet in round 1, pass empty
        msg = self._persona_message(self.get_personality_prompt(), "", base_msg)
        self.memory.add("buyer", opening_price, msg, round_no=1)
        return opening_price, msg

    def respond_to_seller_offer(self, context: NegotiationContext, seller_price: int, seller_message: str) -> Tuple[DealStatus, int, str]:
        parsed = self.observer.parse_price_from_text(seller_message) or seller_price
        self.memory.add("seller", parsed, seller_message, round_no=context.current_round)
        fair_price = self.decision.compute_fair_price(context.product)

        # Accept if fair and within budget
        if seller_price <= fair_price and seller_price <= context.your_budget:
            base_msg = f"That's reasonable. I accept ₹{seller_price:,}. {self.personality['catchphrases'][1]}"
            msg = self._persona_message(self.get_personality_prompt(), seller_message, base_msg)
            self.memory.add("buyer", seller_price, msg, round_no=context.current_round)
            return DealStatus.ACCEPTED, seller_price, msg

        # Near final rounds: accept small premium if within budget
        if context.current_round >= 7 and seller_price <= int(fair_price * 1.05) and seller_price <= context.your_budget:
            base_msg = f"We're close. I can accept ₹{seller_price:,} to wrap this up."
            msg = self._persona_message(self.get_personality_prompt(), seller_message, base_msg)
            self.memory.add("buyer", seller_price, msg, round_no=context.current_round)
            return DealStatus.ACCEPTED, seller_price, msg

        # If above budget: anchor to budget
        if seller_price > context.your_budget:
            counter = context.your_budget
            if context.current_round >= 9:
                base_msg = f"I can't go beyond my budget of ₹{context.your_budget:,}. If you can meet that, we have a deal."
                msg = self._persona_message(self.get_personality_prompt(), seller_message, base_msg)
                self.memory.add("buyer", None, msg, round_no=context.current_round)
                return DealStatus.ONGOING, counter, msg
            else:
                base_msg = f"I can't reach ₹{seller_price:,}. My top is ₹{counter:,}. Let me know."
                msg = self._persona_message(self.get_personality_prompt(), seller_message, base_msg)
                self.memory.add("buyer", counter, msg, round_no=context.current_round)
                return DealStatus.ONGOING, counter, msg

        # Compute counter within budget
        counter_offer = self.decision.propose_counter(context, self.memory)
        counter_offer = min(counter_offer, context.your_budget)

        # If counter >= seller and within budget, accept instead
        if counter_offer >= seller_price and seller_price <= context.your_budget:
            base_msg = f"Good — I can accept ₹{seller_price:,}. Let's close."
            msg = self._persona_message(self.get_personality_prompt(), seller_message, base_msg)
            self.memory.add("buyer", seller_price, msg, round_no=context.current_round)
            return DealStatus.ACCEPTED, seller_price, msg

        # If gap very small, accept
        last_buyer = self.memory.last_buyer_offer() or 0
        gap = seller_price - last_buyer
        if seller_price <= context.your_budget and gap <= max(int(seller_price * 0.02), 1000):
            base_msg = f"We're nearly aligned. I can accept ₹{seller_price:,}."
            msg = self._persona_message(self.get_personality_prompt(), seller_message, base_msg)
            self.memory.add("buyer", seller_price, msg, round_no=context.current_round)
            return DealStatus.ACCEPTED, seller_price, msg

        base_msg = (
            f"{self.personality['catchphrases'][0]} I can go up to ₹{counter_offer:,} right now. "
            f"If you can meet me halfway from your last, we can close swiftly."
        )
        msg = self._persona_message(self.get_personality_prompt(), seller_message, base_msg)
        self.memory.add("buyer", counter_offer, msg, round_no=context.current_round)
        return DealStatus.ONGOING, counter_offer, msg

    def get_personality_prompt(self) -> str:
        return (
            "You are a pragmatic-analytical buyer. Speak calmly, be data-driven, and time-sensitive. "
            f"Use short clear sentences, and optionally include one of these catchphrases: {self.personality['catchphrases']}. "
            "Prioritise quick closes when numbers align and never exceed the stated budget."
        )

    # Diagnostics
    def analyze_negotiation_progress(self, context: NegotiationContext) -> Dict[str, Any]:
        return self.memory.summary()

    def calculate_fair_price(self, product: Product) -> int:
        return self.decision.compute_fair_price(product)


class MockSellerAgent:
    def __init__(self, min_price: int, personality: str = "standard"):
        self.min_price = min_price
        self.personality = personality

    def get_opening_price(self, product: Product) -> Tuple[int, str]:
        price = int(product.base_market_price * 1.5)
        return price, f"These are premium {product.quality_grade} grade {product.name}. I'm asking ₹{price}."

    def respond_to_buyer(self, buyer_offer: int, round_num: int) -> Tuple[int, str, bool]:
        if buyer_offer >= int(self.min_price * 1.1):
            return buyer_offer, f"You have a deal at ₹{buyer_offer}!", True
        if round_num >= 8:
            counter = max(self.min_price, int(buyer_offer * 1.05))
            return counter, f"Final offer: ₹{counter}. Take it or leave it.", False
        else:
            counter = max(self.min_price, int(buyer_offer * 1.15))
            return counter, f"I can come down to ₹{counter}.", False


def run_negotiation_test(buyer_agent: BaseBuyerAgent, product: Product, buyer_budget: int, seller_min: int) -> Dict[str, Any]:
    seller = MockSellerAgent(seller_min)
    context = NegotiationContext(
        product=product,
        your_budget=buyer_budget,
        current_round=0,
        seller_offers=[],
        your_offers=[],
        messages=[]
    )
    seller_price, seller_msg = seller.get_opening_price(product)
    context.seller_offers.append(seller_price)
    context.messages.append({"role": "seller", "message": seller_msg})

    deal_made = False
    final_price = None
    for round_idx in range(10):
        context.current_round = round_idx + 1
        if round_idx == 0:
            buyer_offer, buyer_msg = buyer_agent.generate_opening_offer(context)
            status = DealStatus.ONGOING
        else:
            status, buyer_offer, buyer_msg = buyer_agent.respond_to_seller_offer(
                context, seller_price, seller_msg
            )

        context.your_offers.append(buyer_offer)
        context.messages.append({"role": "buyer", "message": buyer_msg})

        if status == DealStatus.ACCEPTED:
            deal_made = True
            final_price = seller_price
            break

        seller_price, seller_msg, seller_accepts = seller.respond_to_buyer(buyer_offer, round_idx)
        if seller_accepts:
            deal_made = True
            final_price = buyer_offer
            context.messages.append({"role": "seller", "message": seller_msg})
            break

        context.seller_offers.append(seller_price)
        context.messages.append({"role": "seller", "message": seller_msg})

    result = {
        "deal_made": deal_made,
        "final_price": final_price,
        "rounds": context.current_round,
        "savings": (buyer_budget - final_price) if deal_made and final_price is not None else 0,
        "savings_pct": ((buyer_budget - final_price) / buyer_budget * 100) if deal_made and final_price is not None else 0,
        "below_market_pct": ((context.product.base_market_price - final_price) / context.product.base_market_price * 100) if deal_made and final_price is not None else 0,
        "conversation": context.messages
    }
    return result


def run_all_scenarios(use_llm_persona: bool = False):
    scenarios = [
        # Scenario 1: Easy Market
        {
            "product": Product(
                name="Alphonso Mangoes",
                category="Mangoes",
                quantity=100,
                quality_grade="A",
                origin="Ratnagiri",
                base_market_price=180000,
                attributes={"ripeness": "optimal", "export_grade": True}
            ),
            "budget": 200000,
            "seller_min": 150000
        },
        # Scenario 2: Tight Budget
        {
            "product": Product(
                name="Kesar Mangoes",
                category="Mangoes",
                quantity=150,
                quality_grade="B",
                origin="Gujarat",
                base_market_price=150000,
                attributes={"ripeness": "semi-ripe", "export_grade": False}
            ),
            "budget": 140000,
            "seller_min": 125000
        },
        # Scenario 3: Premium
        {
            "product": Product(
                name="Export-Grade Mangoes",
                category="Mangoes",
                quantity=50,
                quality_grade="Export",
                origin="Multiple",
                base_market_price=200000,
                attributes={"ripeness": "export", "export_grade": True}
            ),
            "budget": 190000,
            "seller_min": 175000
        }
    ]

    agent = YourBuyerAgent("ARUNI_Buyer", use_llm_persona=use_llm_persona)
    print("=" * 60)
    print(f"RUNNING SCENARIOS with persona LLM = {use_llm_persona}")
    print(f"Agent: {agent.name} | Personality: {agent.personality['personality_type']}")
    print("=" * 60)

    deals_made = 0
    for i, sc in enumerate(scenarios, 1):
        product = sc["product"]
        budget = sc["budget"]
        seller_min = sc["seller_min"]

        print(f"\nScenario {i}: {product.name}")
        print(f"Market: ₹{product.base_market_price:,} | Budget: ₹{budget:,} | Hidden seller min: ~₹{seller_min:,}")

        result = run_negotiation_test(agent, product, budget, seller_min)
        if result["deal_made"]:
            deals_made += 1
            print(f"✅ DEAL at ₹{result['final_price']:,} in {result['rounds']} rounds")
            print(f"   Savings: ₹{result['savings']:,} ({result['savings_pct']:.1f}%)")
            print(f"   Below Market: {result['below_market_pct']:.1f}%")
        else:
            print(f"❌ NO DEAL after {result['rounds']} rounds")

    print("\n" + "=" * 60)
    print(f"Total Deals: {deals_made}/3")
    print("=" * 60)


if __name__ == "__main__":
    run_all_scenarios(use_llm_persona=True)
