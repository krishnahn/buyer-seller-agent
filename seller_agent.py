import re
import statistics
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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

# ============================================================================
# OLLAMA LANGUAGE MODEL (OPTIONAL)
# ============================================================================

class OllamaLanguageModel:
    """Optional Ollama adapter - only used if enable_llm=True"""
    def __init__(self, model_name: str = "llama3:8b", host: str = "http://localhost:11434"):
        self.model = model_name
        self.host = host.rstrip("/")

    def generate(self, prompt: str, stop: Optional[List[str]] = None, temperature: float = 0.7) -> str:
        try:
            import requests
            payload = {
                "model": self.model,
                "prompt": prompt,
                "options": {"temperature": temperature}
            }
            if stop:
                payload["stop"] = stop
            r = requests.post(f"{self.host}/api/generate", json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            return data.get("response", "").strip()
        except Exception as e:
            print(f"[WARNING] Ollama failed: {e}. Using fallback.")
            return prompt  # Fallback to original message

# ============================================================================
# BUYER AGENT COMPONENTS
# ============================================================================

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
        # Look for ₹123,456 or 123456
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

        # Progressive concession: start small, increase over time
        concession_factor = min(0.15 + (round_no / max_rounds) * 0.45, 0.6)
        target = int(buyer_last + concession_factor * (seller_last - buyer_last))
        target = min(target, context.your_budget)

        # Don't overshoot seller price unless near end
        if round_no < max_rounds - 2 and target > seller_last:
            target = min(seller_last - 1000, context.your_budget)

        # Floor at 50% of market
        floor = int(context.product.base_market_price * 0.5)
        if target < floor:
            target = floor

        return target

# ============================================================================
# BUYER AGENT
# ============================================================================

class YourBuyerAgent:
    def __init__(self, name: str, enable_llm: bool = False):
        self.name = name
        self.personality = self.define_personality()
        self.memory = MemoryComponent()
        self.observer = ObservationComponent()
        self.decision = DecisionComponent(self.personality)
        self.enable_llm = enable_llm
        self.lm = OllamaLanguageModel() if enable_llm else None

    def define_personality(self) -> Dict[str, Any]:
        return {
            "personality_type": "pragmatic-analytical",
            "traits": ["data-driven", "calm", "time-sensitive", "firm"],
            "negotiation_style": "Starts with evidence-based low offers, concedes predictably as rounds progress, and closes promptly when the deal meets fair value or budget constraints.",
            "catchphrases": ["Let's keep this efficient.", "I value clarity and quick closes."]
        }

    def _enhance_message(self, base_message: str, seller_msg: str = "") -> str:
        """Optional LLM enhancement of message tone"""
        if not self.enable_llm or self.lm is None:
            return base_message

        prompt = f"""You are a pragmatic-analytical buyer. Rewrite this message to sound data-driven, calm, and time-sensitive. Keep all numbers exactly the same. Be concise.

Original: {base_message}

Rewritten:"""

        enhanced = self.lm.generate(prompt, temperature=0.3)
        return enhanced if enhanced and len(enhanced) < len(base_message) * 2 else base_message

    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        fair = self.decision.compute_fair_price(context.product)

        # Opening discount by grade
        grade_opening_discount = {"A": 0.30, "B": 0.35, "Export": 0.25}
        discount = grade_opening_discount.get(context.product.quality_grade, 0.32)
        opening_price = int(fair * (1 - discount))
        opening_price = min(opening_price, context.your_budget)

        # Minimum reasonable offer (40% of market)
        min_offer = int(context.product.base_market_price * 0.4)
        if opening_price < min_offer:
            opening_price = min_offer

        base_msg = (
            f"{self.personality['catchphrases'][0]} I can start at ₹{opening_price:,} for "
            f"{context.product.quantity} x {context.product.name}. "
            f"I'm data-driven and prefer a quick close if the numbers make sense."
        )

        msg = self._enhance_message(base_msg)
        self.memory.add("buyer", opening_price, msg, round_no=1)
        return opening_price, msg

    def respond_to_seller_offer(self, context: NegotiationContext, seller_price: int, seller_message: str) -> Tuple[DealStatus, int, str]:
        # Parse and record seller offer
        parsed = self.observer.parse_price_from_text(seller_message) or seller_price
        self.memory.add("seller", parsed, seller_message, round_no=context.current_round)
        fair_price = self.decision.compute_fair_price(context.product)

        # ACCEPTANCE RULES

        # 1) Accept if fair and within budget
        if seller_price <= fair_price and seller_price <= context.your_budget:
            base_msg = f"That's reasonable. I accept ₹{seller_price:,}. {self.personality['catchphrases'][1]}"
            msg = self._enhance_message(base_msg, seller_message)
            self.memory.add("buyer", seller_price, msg, round_no=context.current_round)
            return DealStatus.ACCEPTED, seller_price, msg

        # 2) Near final rounds: accept small premium if within budget
        if context.current_round >= 7 and seller_price <= int(fair_price * 1.05) and seller_price <= context.your_budget:
            base_msg = f"We're close. I can accept ₹{seller_price:,} to wrap this up."
            msg = self._enhance_message(base_msg, seller_message)
            self.memory.add("buyer", seller_price, msg, round_no=context.current_round)
            return DealStatus.ACCEPTED, seller_price, msg

        # 3) If above budget: anchor to budget
        if seller_price > context.your_budget:
            counter = context.your_budget
            if context.current_round >= 9:
                base_msg = f"I can't go beyond my budget of ₹{context.your_budget:,}. If you can meet that, we have a deal."
                msg = self._enhance_message(base_msg, seller_message)
                self.memory.add("buyer", None, msg, round_no=context.current_round)
                return DealStatus.ONGOING, counter, msg
            else:
                base_msg = f"I can't reach ₹{seller_price:,}. My top is ₹{counter:,}. Let me know."
                msg = self._enhance_message(base_msg, seller_message)
                self.memory.add("buyer", counter, msg, round_no=context.current_round)
                return DealStatus.ONGOING, counter, msg

        # COUNTER-OFFER LOGIC

        counter_offer = self.decision.propose_counter(context, self.memory)
        counter_offer = min(counter_offer, context.your_budget)

        # If our counter >= seller price, just accept
        if counter_offer >= seller_price and seller_price <= context.your_budget:
            base_msg = f"Good — I can accept ₹{seller_price:,}. Let's close."
            msg = self._enhance_message(base_msg, seller_message)
            self.memory.add("buyer", seller_price, msg, round_no=context.current_round)
            return DealStatus.ACCEPTED, seller_price, msg

        # If gap is very small (< 2%), accept
        last_buyer = self.memory.last_buyer_offer() or 0
        gap = seller_price - last_buyer
        if seller_price <= context.your_budget and gap <= max(int(seller_price * 0.02), 1000):
            base_msg = f"We're nearly aligned. I can accept ₹{seller_price:,}."
            msg = self._enhance_message(base_msg, seller_message)
            self.memory.add("buyer", seller_price, msg, round_no=context.current_round)
            return DealStatus.ACCEPTED, seller_price, msg

        # Otherwise, send counter
        base_msg = (
            f"{self.personality['catchphrases'][0]} I can go up to ₹{counter_offer:,} right now. "
            f"If you can meet me halfway from your last, we can close swiftly."
        )
        msg = self._enhance_message(base_msg, seller_message)
        self.memory.add("buyer", counter_offer, msg, round_no=context.current_round)
        return DealStatus.ONGOING, counter_offer, msg

# ============================================================================
# ENHANCED SELLER AGENT
# ============================================================================

class EnhancedSellerAgent:
    def __init__(self, min_price: int, personality: str = "standard"):
        self.min_price = min_price
        self.personality = personality

    def get_opening_price(self, product: Product) -> Tuple[int, str]:
        # Personality affects opening strategy
        if self.personality == "aggressive":
            multiplier = 1.6
        elif self.personality == "diplomatic":
            multiplier = 1.4
        else:  # standard
            multiplier = 1.5

        price = int(product.base_market_price * multiplier)

        if self.personality == "diplomatic":
            msg = f"Fresh {product.quality_grade} grade {product.name} from {product.origin}. My opening is ₹{price:,}."
        else:
            msg = f"Premium {product.quality_grade} grade {product.name}. I'm asking ₹{price:,}."

        return price, msg

    def respond_to_buyer(self, buyer_offer: int, round_num: int, product_base_price: int) -> Tuple[int, str, bool]:
        # Accept if buyer meets threshold
        if buyer_offer >= int(self.min_price * 1.08):
            return buyer_offer, f"You have a deal at ₹{buyer_offer:,}!", True

        # Time pressure logic
        if round_num >= 8:
            # Final rounds: get more flexible
            counter = max(self.min_price, int(buyer_offer * 1.04))
            if round_num >= 9:
                msg = f"Final offer: ₹{counter:,}. Take it or leave it."
            else:
                msg = f"I can do ₹{counter:,} to close this."
        else:
            # Early rounds: maintain higher margin
            if self.personality == "aggressive":
                counter = max(self.min_price, int(buyer_offer * 1.18))
            elif self.personality == "diplomatic":
                counter = max(self.min_price, int(buyer_offer * 1.12))
            else:
                counter = max(self.min_price, int(buyer_offer * 1.15))

            # Don't go too high vs market
            ceiling = int(product_base_price * (1.3 if round_num <= 4 else 1.2))
            counter = min(counter, ceiling)

            msg = f"I can come down to ₹{counter:,}."

        return counter, msg, False

# ============================================================================
# SIMULATION RUNNER
# ============================================================================

def run_negotiation_test(buyer_agent, product: Product, buyer_budget: int, seller_min: int, seller_personality: str = "standard") -> Dict[str, Any]:
    print(f"[DEBUG] Starting negotiation: Budget=₹{buyer_budget:,}, Seller Min=₹{seller_min:,}")

    seller = EnhancedSellerAgent(seller_min, seller_personality)
    context = NegotiationContext(
        product=product,
        your_budget=buyer_budget,
        current_round=0,
        seller_offers=[],
        your_offers=[],
        messages=[]
    )

    # Seller opens
    seller_price, seller_msg = seller.get_opening_price(product)
    context.seller_offers.append(seller_price)
    context.messages.append({"role": "seller", "message": seller_msg})
    print(f"[DEBUG] Seller opens: ₹{seller_price:,}")

    deal_made = False
    final_price = None

    for round_idx in range(10):
        context.current_round = round_idx + 1
        print(f"[DEBUG] === Round {context.current_round} ===")

        # Buyer's turn
        if round_idx == 0:
            buyer_offer, buyer_msg = buyer_agent.generate_opening_offer(context)
            status = DealStatus.ONGOING
        else:
            status, buyer_offer, buyer_msg = buyer_agent.respond_to_seller_offer(
                context, seller_price, seller_msg
            )

        context.your_offers.append(buyer_offer)
        context.messages.append({"role": "buyer", "message": buyer_msg})
        print(f"[DEBUG] Buyer: ₹{buyer_offer:,} - {buyer_msg[:50]}...")

        if status == DealStatus.ACCEPTED:
            deal_made = True
            final_price = seller_price
            print(f"[DEBUG] Buyer accepted at ₹{seller_price:,}")
            break

        # Seller's turn
        seller_price, seller_msg, seller_accepts = seller.respond_to_buyer(
            buyer_offer, round_idx, product.base_market_price
        )
        print(f"[DEBUG] Seller: ₹{seller_price:,} - {seller_msg[:50]}...")

        if seller_accepts:
            deal_made = True
            final_price = buyer_offer
            context.messages.append({"role": "seller", "message": seller_msg})
            print(f"[DEBUG] Seller accepted at ₹{buyer_offer:,}")
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

def run_all_scenarios(enable_llm: bool = False):
    print("=" * 80)
    print("🥭 MANGO NEGOTIATION SIMULATION")
    print(f"LLM Enhancement: {'ENABLED' if enable_llm else 'DISABLED'}")
    print("=" * 80)

    scenarios = [
        {
            "name": "Easy Market",
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
            "seller_min": 150000,
            "seller_personality": "diplomatic"
        },
        {
            "name": "Tight Budget",
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
            "seller_min": 125000,
            "seller_personality": "standard"
        },
        {
            "name": "Premium Product",
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
            "seller_min": 175000,
            "seller_personality": "aggressive"
        }
    ]

    agent = YourBuyerAgent("ARUNI_Buyer", enable_llm=enable_llm)
    print(f"Agent: {agent.name}")
    print(f"Personality: {agent.personality['personality_type']}")
    print()

    deals_made = 0
    total_savings = 0

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🎯 SCENARIO {i}: {scenario['name']}")
        print(f"Product: {scenario['product'].name}")
        print(f"Market: ₹{scenario['product'].base_market_price:,} | Budget: ₹{scenario['budget']:,}")
        print(f"Seller Min: ~₹{scenario['seller_min']:,} | Seller Type: {scenario['seller_personality']}")
        print("-" * 60)

        result = run_negotiation_test(
            agent,
            scenario["product"],
            scenario["budget"],
            scenario["seller_min"],
            scenario["seller_personality"]
        )

        if result["deal_made"]:
            deals_made += 1
            total_savings += result["savings"]
            print(f"✅ SUCCESS: Deal closed at ₹{result['final_price']:,} in {result['rounds']} rounds")
            print(f"   💰 Savings: ₹{result['savings']:,} ({result['savings_pct']:.1f}% of budget)")
            print(f"   📊 Below Market: {result['below_market_pct']:.1f}%")
        else:
            print(f"❌ FAILED: No deal after {result['rounds']} rounds")

        print("-" * 60)

    print(f"\n🏆 FINAL RESULTS")
    print(f"Deals Completed: {deals_made}/3")
    print(f"Success Rate: {deals_made/3*100:.1f}%")
    print(f"Total Savings: ₹{total_savings:,}")
    print("=" * 80)

if __name__ == "__main__":
    print("Starting negotiation simulation...")

    # Set to True if you want LLM enhancement and have Ollama running
    # Set to False for pure logic (no LLM dependency)
    USE_LLM = False

    run_all_scenarios(enable_llm=USE_LLM)
    print("Simulation complete!")
