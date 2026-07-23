"""
M1 dataset — Knowledge Gap Classifier — STARTER question bank.

IMPORTANT — this is a starter set, not the final dataset. Per
MENTORA_Phase3_Datasets.md §4.6, the target is >=150-300 tagged
questions per subject; this script ships ~15-25 per subject (80 total)
so the M1/M2 pipeline is runnable end-to-end today. Before Phase 4
training, scale this up with real past-paper questions (§4.2-4.3) —
every question below is original (written for this project, not
copied from any past paper or Kaggle set), so there's no provenance
overlap to worry about when you add real sourced questions later.

Taxonomy (must match exam-prep.html's subject chips and
TOPIC_SCORES.subject exactly — see §7 cross-consistency check):
  Mathematics: Algebra, Geometry, Trigonometry, Calculus, Statistics
  English: Grammar, Reading Comprehension, Vocabulary, Writing
  Physics: Mechanics, Electricity & Magnetism, Waves & Optics, Modern Physics
  Chemistry: Organic, Inorganic, Physical Chemistry

Run: python3 generate_m1_starter_bank.py
Output: ../processed/m1/question_bank.csv
"""
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "processed" / "m1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Each tuple: (subject, topic, difficulty, question, A, B, C, D, correct)
QUESTIONS = [
    # --- Mathematics: Algebra ---
    ("Mathematics", "Algebra", "easy", "Solve for x: 2x + 6 = 14", "3", "4", "5", "8", "B"),
    ("Mathematics", "Algebra", "easy", "Simplify: 3x + 5x", "8x", "15x", "8x^2", "2x", "A"),
    ("Mathematics", "Algebra", "medium", "Solve x^2 - 5x + 6 = 0. What are the roots?", "x = 2, 3", "x = -2, 3", "x = 1, 6", "x = -1, -6", "A"),
    ("Mathematics", "Algebra", "medium", "Factorize: x^2 - 9", "(x-3)(x+3)", "(x-9)(x+1)", "(x-3)^2", "(x+9)(x-1)", "A"),
    ("Mathematics", "Algebra", "hard", "If 2^x = 32, what is the value of x?", "4", "5", "6", "16", "B"),
    ("Mathematics", "Algebra", "medium", "Solve the system: x + y = 10, x - y = 2. Find x.", "4", "5", "6", "8", "C"),
    # --- Mathematics: Geometry ---
    ("Mathematics", "Geometry", "easy", "What is the sum of interior angles of a triangle?", "90 degrees", "180 degrees", "270 degrees", "360 degrees", "B"),
    ("Mathematics", "Geometry", "easy", "A square has a side of 5 cm. What is its area?", "10 sq cm", "20 sq cm", "25 sq cm", "30 sq cm", "C"),
    ("Mathematics", "Geometry", "medium", "The circumference of a circle with radius 7 cm is (use pi = 22/7)", "22 cm", "44 cm", "14 cm", "88 cm", "B"),
    ("Mathematics", "Geometry", "medium", "In a right triangle, the two legs are 3 cm and 4 cm. What is the hypotenuse?", "5 cm", "6 cm", "7 cm", "12 cm", "A"),
    ("Mathematics", "Geometry", "hard", "A cylinder has radius 3 cm and height 7 cm. What is its volume (use pi = 22/7)?", "154 cu cm", "198 cu cm", "132 cu cm", "231 cu cm", "C"),
    # --- Mathematics: Trigonometry ---
    ("Mathematics", "Trigonometry", "easy", "What is sin(90 degrees)?", "0", "0.5", "1", "undefined", "C"),
    ("Mathematics", "Trigonometry", "medium", "What is the value of cos(60 degrees)?", "1", "0.5", "0.87", "0", "B"),
    ("Mathematics", "Trigonometry", "medium", "If tan(theta) = 1, what is theta (0-90 degrees range)?", "30 degrees", "45 degrees", "60 degrees", "90 degrees", "B"),
    ("Mathematics", "Trigonometry", "hard", "Simplify: sin^2(theta) + cos^2(theta)", "0", "1", "2", "tan(theta)", "B"),
    # --- Mathematics: Calculus ---
    ("Mathematics", "Calculus", "medium", "What is the derivative of x^3?", "3x^2", "x^2", "3x", "x^3/3", "A"),
    ("Mathematics", "Calculus", "medium", "What is the integral of 2x dx?", "x^2 + C", "2x^2 + C", "x^2/2 + C", "2 + C", "A"),
    ("Mathematics", "Calculus", "hard", "What is the derivative of sin(x)?", "cos(x)", "-cos(x)", "-sin(x)", "tan(x)", "A"),
    ("Mathematics", "Calculus", "hard", "Evaluate the limit as x approaches 0 of (sin x)/x", "0", "1", "infinity", "undefined", "B"),
    # --- Mathematics: Statistics ---
    ("Mathematics", "Statistics", "easy", "What is the mean of 2, 4, 6, 8, 10?", "5", "6", "7", "8", "B"),
    ("Mathematics", "Statistics", "medium", "What is the median of 3, 7, 9, 15, 20?", "7", "9", "15", "10.8", "B"),
    ("Mathematics", "Statistics", "medium", "What does standard deviation measure?", "Central tendency", "Data spread", "Sample size", "Correlation", "B"),
    ("Mathematics", "Statistics", "hard", "In a normal distribution, approximately what percent of data falls within 1 standard deviation of the mean?", "50 percent", "68 percent", "95 percent", "99 percent", "B"),

    # --- English: Grammar ---
    ("English", "Grammar", "easy", "Choose the correct verb: She ___ to school every day.", "go", "goes", "going", "gone", "B"),
    ("English", "Grammar", "easy", "Identify the correct sentence.", "He don't like tea.", "He doesn't likes tea.", "He doesn't like tea.", "He not like tea.", "C"),
    ("English", "Grammar", "medium", "Choose the correct passive form: 'They built the house in 1990.'", "The house built in 1990.", "The house was built in 1990.", "The house is built in 1990.", "The house has build in 1990.", "B"),
    ("English", "Grammar", "medium", "Which sentence uses the correct conditional form? 'If it rains, ___'", "I will stayed home.", "I would stay home.", "I will stay home.", "I stay home.", "C"),
    ("English", "Grammar", "hard", "Choose the sentence with correct subject-verb agreement.", "Neither of the boys were present.", "Neither of the boys was present.", "Neither of the boys are present.", "Neither of the boys be present.", "B"),
    # --- English: Reading Comprehension ---
    ("English", "Reading Comprehension", "medium", "A passage states 'Despite the heavy rain, the match continued.' This implies the match:", "was cancelled", "was delayed", "continued despite bad weather", "was moved indoors", "C"),
    ("English", "Reading Comprehension", "medium", "In comprehension passages, the 'main idea' usually refers to:", "A minor detail", "The central point of the passage", "The last sentence only", "An example used in the text", "B"),
    ("English", "Reading Comprehension", "hard", "If a passage uses the word 'however' between two sentences, this signals:", "Addition", "Contrast", "Cause and effect", "Summary", "B"),
    ("English", "Reading Comprehension", "hard", "Identifying an author's tone requires attention to:", "Only the topic", "Word choice and phrasing", "Paragraph length", "Number of sentences", "B"),
    # --- English: Vocabulary ---
    ("English", "Vocabulary", "easy", "Choose the synonym of 'Happy'.", "Sad", "Joyful", "Angry", "Tired", "B"),
    ("English", "Vocabulary", "easy", "Choose the antonym of 'Ancient'.", "Old", "Modern", "Historic", "Aged", "B"),
    ("English", "Vocabulary", "medium", "Choose the correct meaning of 'Ambiguous'.", "Very clear", "Open to more than one interpretation", "Extremely loud", "Perfectly accurate", "B"),
    ("English", "Vocabulary", "medium", "Choose the synonym of 'Diligent'.", "Lazy", "Hardworking", "Careless", "Slow", "B"),
    ("English", "Vocabulary", "hard", "Choose the correct meaning of 'Ephemeral'.", "Lasting forever", "Short-lived", "Very strong", "Extremely rare", "B"),
    # --- English: Writing ---
    ("English", "Writing", "medium", "Which of the following is a compound sentence?", "She ran fast.", "She ran fast, and she won the race.", "Running fast, she won.", "The fast runner.", "B"),
    ("English", "Writing", "medium", "Choose the correctly punctuated sentence.", "I bought apples oranges and bananas.", "I bought apples, oranges, and bananas.", "I bought apples oranges, and bananas.", "I bought, apples oranges and bananas.", "B"),
    ("English", "Writing", "hard", "Which sentence avoids a dangling modifier?", "Walking to school, the rain started.", "Walking to school, I got caught in the rain.", "The rain, walking to school, started.", "Walking, the school and rain.", "B"),

    # --- Physics: Mechanics ---
    ("Physics", "Mechanics", "easy", "What is the SI unit of force?", "Joule", "Newton", "Watt", "Pascal", "B"),
    ("Physics", "Mechanics", "easy", "An object at rest stays at rest unless acted upon by an external force. This is:", "Newton's First Law", "Newton's Second Law", "Newton's Third Law", "Law of Gravitation", "A"),
    ("Physics", "Mechanics", "medium", "A force of 10 N acts on a mass of 2 kg. What is the acceleration?", "2 m/s^2", "5 m/s^2", "10 m/s^2", "20 m/s^2", "B"),
    ("Physics", "Mechanics", "medium", "What is the formula for kinetic energy?", "mgh", "1/2 mv^2", "mv", "F x d", "B"),
    ("Physics", "Mechanics", "hard", "A ball is thrown upward with initial velocity 20 m/s. Ignoring air resistance, how long does it take to reach maximum height? (g = 10 m/s^2)", "1 s", "2 s", "3 s", "4 s", "B"),
    # --- Physics: Electricity & Magnetism ---
    ("Physics", "Electricity & Magnetism", "easy", "What is the SI unit of electric current?", "Volt", "Ohm", "Ampere", "Watt", "C"),
    ("Physics", "Electricity & Magnetism", "medium", "According to Ohm's Law, V = IR. If I = 2A and R = 5 ohms, what is V?", "2.5 V", "7 V", "10 V", "20 V", "C"),
    ("Physics", "Electricity & Magnetism", "medium", "Which particle carries negative charge?", "Proton", "Neutron", "Electron", "Positron", "C"),
    ("Physics", "Electricity & Magnetism", "hard", "Two resistors of 4 ohms and 6 ohms are connected in series. What is the total resistance?", "2.4 ohms", "10 ohms", "24 ohms", "1.5 ohms", "B"),
    # --- Physics: Waves & Optics ---
    ("Physics", "Waves & Optics", "easy", "What type of wave is sound?", "Transverse", "Longitudinal", "Electromagnetic", "Standing", "B"),
    ("Physics", "Waves & Optics", "medium", "The speed of light in vacuum is approximately:", "3 x 10^5 m/s", "3 x 10^6 m/s", "3 x 10^8 m/s", "3 x 10^10 m/s", "C"),
    ("Physics", "Waves & Optics", "medium", "A converging lens is also known as a:", "Concave lens", "Convex lens", "Plane lens", "Diverging lens", "B"),
    ("Physics", "Waves & Optics", "hard", "If wave frequency is 50 Hz and wavelength is 2 m, what is the wave speed?", "25 m/s", "52 m/s", "100 m/s", "0.04 m/s", "C"),
    # --- Physics: Modern Physics ---
    ("Physics", "Modern Physics", "medium", "What particle is emitted in beta decay?", "Alpha particle", "Electron", "Neutron", "Photon", "B"),
    ("Physics", "Modern Physics", "medium", "Einstein's mass-energy equivalence formula is:", "E = mc", "E = mc^2", "E = m/c^2", "E = 2mc", "B"),
    ("Physics", "Modern Physics", "hard", "The photoelectric effect provided evidence for:", "Wave nature of light only", "Particle nature of light (photons)", "Sound as a particle", "Gravity waves", "B"),

    # --- Chemistry: Organic ---
    ("Chemistry", "Organic", "easy", "What is the general formula for alkanes?", "CnH2n", "CnH2n+2", "CnH2n-2", "CnHn", "B"),
    ("Chemistry", "Organic", "easy", "Methane's chemical formula is:", "CH2", "CH4", "C2H6", "CH3", "B"),
    ("Chemistry", "Organic", "medium", "Which functional group is present in alcohols?", "-COOH", "-OH", "-CHO", "-NH2", "B"),
    ("Chemistry", "Organic", "medium", "Ethene (C2H4) is an example of which hydrocarbon class?", "Alkane", "Alkene", "Alkyne", "Aromatic", "B"),
    ("Chemistry", "Organic", "hard", "What type of reaction converts an alkene to an alkane?", "Oxidation", "Hydrogenation", "Dehydration", "Substitution", "B"),
    # --- Chemistry: Inorganic ---
    ("Chemistry", "Inorganic", "easy", "What is the chemical symbol for Sodium?", "So", "Sd", "Na", "S", "C"),
    ("Chemistry", "Inorganic", "easy", "How many electrons does a neutral Oxygen atom (atomic number 8) have?", "6", "7", "8", "9", "C"),
    ("Chemistry", "Inorganic", "medium", "Which group of the periodic table contains the noble gases?", "Group 1", "Group 7", "Group 17", "Group 18", "D"),
    ("Chemistry", "Inorganic", "medium", "What is the charge on a calcium ion (Ca) when it forms an ionic bond?", "+1", "+2", "-1", "-2", "B"),
    ("Chemistry", "Inorganic", "hard", "Which of these is an example of an amphoteric oxide?", "Na2O", "Al2O3", "CO2", "MgO", "B"),
    # --- Chemistry: Physical Chemistry ---
    ("Chemistry", "Physical Chemistry", "easy", "What is the SI unit of temperature used in gas law calculations?", "Celsius", "Fahrenheit", "Kelvin", "Rankine", "C"),
    ("Chemistry", "Physical Chemistry", "medium", "According to the Ideal Gas Law PV = nRT, if volume and temperature are constant and moles increase, pressure will:", "Decrease", "Increase", "Stay the same", "Become zero", "B"),
    ("Chemistry", "Physical Chemistry", "medium", "What is the pH of a neutral solution at 25 degrees C?", "0", "7", "14", "1", "B"),
    ("Chemistry", "Physical Chemistry", "hard", "In an exothermic reaction, the enthalpy change (delta H) is:", "Positive", "Negative", "Zero", "Undefined", "B"),
]

rows = []
for i, (subject, topic, difficulty, q, a, b, c, d, correct) in enumerate(QUESTIONS, start=1):
    rows.append({
        "question_id": f"Q{1000 + i}",
        "question_text": q,
        "subject": subject,
        "topic": topic,
        "subtopic": None,
        "difficulty": difficulty,
        "option_a": a, "option_b": b, "option_c": c, "option_d": d,
        "correct_answer": correct,
        "source": "hand_authored_starter",
    })

df = pd.DataFrame(rows)
assert df["question_text"].is_unique, "Duplicate question text found"

# Validation checklist from §4.6 (informational — this starter set is
# intentionally below the >=150-300/subject target; see the module docstring)
print("Questions per subject:\n", df["subject"].value_counts())
print("\nQuestions per topic:\n", df.groupby(["subject", "topic"]).size())

OUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_DIR / "question_bank.csv", index=False)
print(f"\nWrote {len(df)} questions -> {OUT_DIR / 'question_bank.csv'}")
