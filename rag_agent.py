"""
DIP Diet RAG Agent - IBM Granite-powered conversational backend.
Combines ChromaDB vector retrieval with IBM watsonx.ai Granite LLM
to deliver personalized DIP diet recommendations.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

# ── watsonx.ai credentials (overridden by .env / env vars) ─────────────────
WATSONX_API_KEY    = os.environ.get("WATSONX_API_KEY",    "MnJ6OgqqjjSptm7b4mBz3CR-pHKfjl0ZizxdZxfjBhac")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "fbe24a94-8963-456a-8483-53f67679d829")
WATSONX_URL        = os.environ.get("WATSONX_URL",        "https://us-south.ml.cloud.ibm.com")
GRANITE_MODEL      = os.environ.get("GRANITE_MODEL",      "meta-llama/llama-3-3-70b-instruct")


# ── User profile dataclass ──────────────────────────────────────────────────
@dataclass
class UserProfile:
    name: str = "User"
    age: int = 30
    gender: str = "Not specified"
    weight_kg: float = 70.0
    height_cm: float = 170.0
    activity_level: str = "Moderate"
    health_goals: str = "General wellness"
    medical_conditions: str = "None"
    dietary_restrictions: str = "None"

    @property
    def bmi(self) -> float:
        if self.height_cm <= 0:
            return 0.0
        return round(self.weight_kg / ((self.height_cm / 100) ** 2), 1)

    @property
    def bmi_category(self) -> str:
        b = self.bmi
        if b < 18.5:
            return "Underweight"
        elif b < 25:
            return "Normal weight"
        elif b < 30:
            return "Overweight"
        return "Obese"

    def summary(self) -> str:
        return (
            f"Name: {self.name}\n"
            f"Age: {self.age}\n"
            f"Gender: {self.gender}\n"
            f"BMI: {self.bmi} ({self.bmi_category})\n"
            f"Activity Level: {self.activity_level}\n"
            f"Health Goals: {self.health_goals}\n"
            f"Medical Conditions: {self.medical_conditions}\n"
            f"Dietary Restrictions: {self.dietary_restrictions}"
        )


# ── LLM initializer ─────────────────────────────────────────────────────────
def _init_llm():
    """Initialize IBM Granite LLM via watsonx.ai."""
    from ibm_watsonx_ai import APIClient, Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference

    creds = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
    client = APIClient(creds)
    model = ModelInference(
        model_id=GRANITE_MODEL,
        api_client=client,
        project_id=WATSONX_PROJECT_ID,
        params={
            "max_new_tokens": 1024,
            "temperature": 0.3,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        },
    )
    return model


# ── RAG Agent ───────────────────────────────────────────────────────────────
class DIPDietRAGAgent:
    """
    Retrieval-Augmented Generation agent for DIP Diet.
    Uses ChromaDB for semantic retrieval and IBM Granite for generation.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.retriever = None
        self.llm = None
        self.status = {}

    def initialize(self) -> dict:
        """Initialize retriever and LLM. Returns status dict."""
        from ingest import get_retriever

        status = {"retriever": None, "llm": None, "retriever_chunks": 0, "errors": []}

        # Retriever
        try:
            self.retriever = get_retriever(top_k=self.top_k)
            test = self.retriever("DIP diet breakfast")
            status["retriever"] = f"Retriever: OK ({len(test)} chunks)"
            status["retriever_chunks"] = len(test)
        except Exception as e:
            status["errors"].append(f"Retriever: {e}")

        # LLM
        try:
            self.llm = _init_llm()
            status["llm"] = "LLM: IBM Granite (watsonx.ai)"
        except Exception as e:
            status["errors"].append(f"LLM: {e}")
            self.llm = None

        self.status = status
        return status

    # ── Prompt builder ───────────────────────────────────────────────────────
    def _build_prompt(
        self,
        question: str,
        context_chunks: list[dict],
        user_profile: Optional[UserProfile] = None,
    ) -> str:
        """Build a structured prompt compatible with Llama-3 and Granite instruct models."""
        context = "\n\n---\n\n".join(
            f"[Source: {c.get('source', 'unknown')}]\n{c.get('content', '')}"
            for c in context_chunks
        )
        profile_text = user_profile.summary() if user_profile else "Not provided"

        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are an expert DIP Diet assistant. The DIP Diet (Disciplined & Intelligent Plan) "
            "is a plant-based whole-food dietary system developed by Dr. Biswaroop Roy Chowdhury. "
            "Your role is to provide accurate, personalized, and actionable dietary guidance strictly "
            "based on DIP principles.\n\n"
            "Guidelines:\n"
            "- Always base your answers on the provided context from the DIP Diet knowledge base.\n"
            "- Personalize advice based on the user's health profile when relevant.\n"
            "- Be concise, clear, and supportive.\n"
            "- Never recommend non-DIP foods unless as a substitution note.\n"
            "- Include specific food recommendations, portion sizes, and timing when relevant.\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"USER HEALTH PROFILE:\n{profile_text}\n\n"
            f"RELEVANT DIP DIET KNOWLEDGE:\n{context}\n\n"
            f"QUESTION: {question}"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    # ── Query pipeline ───────────────────────────────────────────────────────
    def query(
        self,
        question: str,
        user_profile: Optional[UserProfile] = None,
        chat_history: Optional[list] = None,
    ) -> dict:
        """
        Process a user question through the RAG pipeline.
        Returns answer, sources, and retrieval scores.
        """
        result = {"answer": "", "sources": [], "error": None, "retrieved_chunks": []}

        # Retrieve
        try:
            chunks = self.retriever(question) if self.retriever else []
            result["retrieved_chunks"] = chunks
            result["sources"] = list({c.get("source", "unknown") for c in chunks})
        except Exception as e:
            result["error"] = f"Retrieval failed: {e}"
            result["answer"] = self._fallback_answer([])
            return result

        # Generate
        if self.llm:
            try:
                prompt = self._build_prompt(question, chunks, user_profile)
                response = self.llm.generate(prompt)
                # ibm_watsonx_ai returns dict with results list
                if isinstance(response, dict):
                    results = response.get("results", [{}])
                    result["answer"] = results[0].get("generated_text", "").strip()
                else:
                    result["answer"] = str(response).strip()

                if result["sources"]:
                    result["answer"] += f"\n\n*Sources: {', '.join(result['sources'])}*"
            except Exception as e:
                result["error"] = f"Generation failed: {e}"
                result["answer"] = self._fallback_answer(chunks)
        else:
            result["error"] = f"LLM unavailable: {self.status.get('errors', [])}"
            result["answer"] = self._fallback_answer(chunks)

        return result

    # ── Fallback ─────────────────────────────────────────────────────────────
    @staticmethod
    def _fallback_answer(chunks: list[dict]) -> str:
        """Fallback: compile answer directly from retrieved context chunks."""
        if not chunks:
            return (
                "I'm unable to retrieve specific information right now. "
                "Based on DIP Diet principles: focus on fruits for breakfast, "
                "50% raw salad + 50% cooked whole food for lunch and dinner, "
                "avoid all dairy, processed foods, and refined grains. "
                "Please ensure the vector store is initialized by running ingest.py."
            )
        main = chunks[0].get("content", "")
        extra_sources = ", ".join(
            c.get("source", "unknown") for c in chunks[1:] if c.get("source")
        )
        answer = f"**Based on DIP Diet knowledge base:**\n\n{main}"
        if extra_sources:
            answer += f"\n\n*(Additional context from: {extra_sources})*"
        return answer


# ── Meal plan generator ──────────────────────────────────────────────────────
def generate_meal_plan(user_profile: Optional[UserProfile] = None) -> dict:
    """
    Generate a 7-day meal plan dict adapted to user profile.
    Returns structured data for dashboard visualization.
    """
    base_plan = {
        "Monday": {
            "breakfast": "Watermelon + Banana + Apple",
            "lunch_salad": "Cucumber + Tomato + Carrot + Onion (lemon dressing)",
            "lunch_cooked": "Brown rice + Yellow moong dal",
            "dinner_salad": "Beetroot + Capsicum + Radish (lemon dressing)",
            "dinner_cooked": "Bajra roti + Bottle gourd sabzi",
            "snack": "5 soaked almonds + 2 walnuts",
            "calories": 1580,
        },
        "Tuesday": {
            "breakfast": "Papaya + Pomegranate + Guava",
            "lunch_salad": "Cabbage + Tomato + Cucumber + Grated carrot",
            "lunch_cooked": "Jowar roti + Chana dal",
            "dinner_salad": "Spinach + Beetroot + Tomato",
            "dinner_cooked": "Moong-rice khichdi",
            "snack": "Coconut water + fresh coconut",
            "calories": 1620,
        },
        "Wednesday": {
            "breakfast": "Mango + Banana + Grapes",
            "lunch_salad": "Radish + Carrot + Capsicum + Coriander",
            "lunch_cooked": "Ragi roti + Toor dal + Drumstick",
            "dinner_salad": "Cucumber + Tomato + Onion",
            "dinner_cooked": "Foxtail millet khichdi",
            "snack": "Mixed sprouts with lemon",
            "calories": 1660,
        },
        "Thursday": {
            "breakfast": "Orange + Apple + Pear",
            "lunch_salad": "Cabbage + Beetroot + Cucumber",
            "lunch_cooked": "Brown rice + Rajma",
            "dinner_salad": "Carrot + Radish + Tomato",
            "dinner_cooked": "Whole wheat roti + Mixed veg sabzi",
            "snack": "Chia seeds in water",
            "calories": 1590,
        },
        "Friday": {
            "breakfast": "Papaya + Watermelon + Pomegranate",
            "lunch_salad": "Spinach + Cucumber + Tomato + Onion",
            "lunch_cooked": "Daliya + Moong dal",
            "dinner_salad": "Capsicum + Carrot + Beetroot",
            "dinner_cooked": "Bajra khichdi",
            "snack": "Fresh seasonal fruit",
            "calories": 1550,
        },
        "Saturday": {
            "breakfast": "Mixed fruit bowl (5-6 varieties)",
            "lunch_salad": "Raw cabbage + Grated carrot + Cucumber",
            "lunch_cooked": "Brown rice + Masoor dal + Brinjal",
            "dinner_salad": "Tomato + Onion + Capsicum + Coriander",
            "dinner_cooked": "Jowar roti + Bottle gourd curry",
            "snack": "Soaked almonds + flaxseed water",
            "calories": 1640,
        },
        "Sunday": {
            "breakfast": "Banana + Apple + Seasonal fruits",
            "lunch_salad": "Spinach + Beetroot + Radish",
            "lunch_cooked": "Vegetable khichdi (mixed veg + moong + rice)",
            "dinner_salad": "Spinach + Beetroot + Radish",
            "dinner_cooked": "Ragi roti + Lauki + chana sabzi",
            "snack": "Coconut water",
            "calories": 1600,
        },
    }

    # Personalize based on profile
    if user_profile:
        goals_lower = user_profile.health_goals.lower()
        conditions_lower = user_profile.medical_conditions.lower()

        for day, meals in base_plan.items():
            # Diabetes: swap high-GI fruits
            if "diabetes" in conditions_lower:
                if meals["breakfast"] in ("Mango + Banana + Grapes",):
                    meals["breakfast"] = "Guava + Apple + Pear"
                meals["notes"] = meals.get("notes", "") + " | Low-GI fruit selection for diabetes management"

            # Hypertension: flag salt
            if "hypertension" in conditions_lower or "blood pressure" in conditions_lower:
                meals["notes"] = meals.get("notes", "") + " | Minimal salt, increase coconut water"

            # Weight loss: bigger salad portions
            if "weight loss" in goals_lower:
                meals["lunch_salad"] = "Double portion: " + meals["lunch_salad"]
                meals["calories"] = int(meals["calories"] * 0.92)

    return base_plan


# ── Nutrition profile ────────────────────────────────────────────────────────
def get_nutrition_profile() -> dict:
    """Return DIP diet nutrition profile for charts."""
    return {
        "macros": {
            "Carbohydrates": 65,
            "Protein": 15,
            "Fats": 20,
        },
        "meal_calories": {
            "Breakfast (Fruits)": 320,
            "Lunch Salad": 180,
            "Lunch Cooked": 480,
            "Dinner Salad": 160,
            "Dinner Cooked": 420,
            "Snacks": 100,
        },
        "weekly_avg_calories": [1580, 1620, 1660, 1590, 1550, 1640, 1600],
        "nutrient_scores": {
            "Fiber": 95,
            "Vitamins": 90,
            "Minerals": 88,
            "Antioxidants": 95,
            "Protein": 72,
            "Healthy Fats": 80,
        },
    }
