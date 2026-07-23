"""
M4 dataset — Career-Skill Semantic Matcher — STARTER career profiles.

IMPORTANT: the plan's real source is O*NET (Occupation Data + Skills flat
files) filtered to ~50-80 CS/tech/business-relevant careers (§3.2). O*NET's
own download site isn't reachable from this build environment, so this
script ships a hand-authored starter set covering the same ~50-career
neighborhood (Data Scientist, ML Engineer, and similar) with realistic
skill lists, so M4's training pipeline is runnable today.

Before Phase 4 training, consider swapping this for a real O*NET join
using the §3.2 script (occ.merge(skills...)) for broader coverage and
defensible data provenance in the thesis — this starter set is original
content written for this project, not scraped or copied from O*NET, so
there's no need to "cite" it, but it also doesn't carry O*NET's breadth
or the specific-skill-importance scoring O*NET provides.

Run: python3 generate_m4_starter_careers.py
Output: ../processed/m4/career_profiles.csv
        ../processed/m4/training_pairs.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "processed" / "m4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (title, required_skills[list], description)
CAREERS = [
    ("Data Scientist", ["Python", "Statistics", "Machine Learning", "SQL", "Data Visualization"], "Analyzes complex data to help organizations make decisions using statistical and ML methods."),
    ("Machine Learning Engineer", ["Python", "TensorFlow", "Deep Learning", "MLOps", "Software Engineering"], "Builds and deploys machine learning models into production systems."),
    ("Data Analyst", ["Excel", "SQL", "Data Visualization", "Statistics", "Tableau"], "Collects and interprets data to answer business questions and build reports."),
    ("Business Analyst", ["SQL", "Excel", "Requirements Gathering", "Process Mapping", "Communication"], "Bridges business needs and technical solutions, analyzing processes and requirements."),
    ("Software Engineer", ["Programming", "Data Structures", "Algorithms", "System Design", "Version Control"], "Designs, builds, and maintains software applications and systems."),
    ("Backend Developer", ["APIs", "Databases", "Server Architecture", "Python", "Node.js"], "Builds server-side logic, databases, and APIs that power applications."),
    ("Frontend Developer", ["JavaScript", "React", "HTML/CSS", "UI Design", "Responsive Design"], "Builds user-facing interfaces and interactive web experiences."),
    ("Full Stack Developer", ["JavaScript", "React", "Node.js", "Databases", "APIs"], "Works across both frontend and backend layers of web applications."),
    ("DevOps Engineer", ["CI/CD", "Docker", "Kubernetes", "Cloud Platforms", "Scripting"], "Automates deployment pipelines and manages infrastructure reliability."),
    ("Cloud Engineer", ["AWS", "Azure", "Networking", "Infrastructure as Code", "Security"], "Designs and manages cloud-based infrastructure and services."),
    ("Cybersecurity Analyst", ["Network Security", "Risk Assessment", "Incident Response", "Firewalls", "Penetration Testing"], "Protects systems and networks from security threats and breaches."),
    ("Database Administrator", ["SQL", "Database Design", "Backup and Recovery", "Performance Tuning", "Security"], "Manages, secures, and optimizes an organization's databases."),
    ("Product Manager", ["Strategy", "User Research", "Roadmapping", "Communication", "Prioritization"], "Defines product vision and coordinates teams to build and ship products."),
    ("Project Manager", ["Planning", "Risk Management", "Communication", "Scheduling", "Budgeting"], "Plans and oversees projects from initiation through completion."),
    ("UX Designer", ["User Research", "Wireframing", "Prototyping", "Usability Testing", "Figma"], "Designs user experiences that are intuitive, accessible, and effective."),
    ("UI Designer", ["Visual Design", "Figma", "Typography", "Color Theory", "Design Systems"], "Designs the visual look and feel of digital products."),
    ("QA Engineer", ["Test Planning", "Automation Testing", "Bug Tracking", "Selenium", "Attention to Detail"], "Ensures software quality through systematic testing and validation."),
    ("Systems Analyst", ["Requirements Analysis", "Systems Design", "Documentation", "SQL", "Process Improvement"], "Analyzes and designs information systems to meet organizational needs."),
    ("Network Engineer", ["Networking Protocols", "Routing and Switching", "Firewalls", "Troubleshooting", "Cisco"], "Designs and maintains computer networks and connectivity."),
    ("IT Support Specialist", ["Troubleshooting", "Customer Service", "Hardware", "Operating Systems", "Ticketing Systems"], "Provides technical support and resolves IT issues for users."),
    ("Data Engineer", ["ETL Pipelines", "SQL", "Python", "Big Data Tools", "Data Warehousing"], "Builds and maintains data pipelines and infrastructure for analytics."),
    ("AI Research Scientist", ["Deep Learning", "Research Methods", "Python", "Mathematics", "Academic Writing"], "Conducts research to advance the state of the art in AI/ML."),
    ("NLP Engineer", ["Natural Language Processing", "Python", "Transformers", "Linguistics", "Machine Learning"], "Builds systems that understand and generate human language."),
    ("Computer Vision Engineer", ["Image Processing", "Deep Learning", "OpenCV", "Python", "Mathematics"], "Builds systems that interpret and analyze visual data."),
    ("Mobile App Developer", ["Swift", "Kotlin", "Mobile UI Design", "APIs", "App Store Deployment"], "Builds native or cross-platform mobile applications."),
    ("Game Developer", ["Unity", "C#", "3D Modeling", "Physics Simulation", "Game Design"], "Designs and builds interactive games and simulations."),
    ("Technical Writer", ["Documentation", "Communication", "Research", "Editing", "Basic Technical Knowledge"], "Writes clear technical documentation and user guides."),
    ("Solutions Architect", ["System Design", "Cloud Platforms", "Enterprise Architecture", "Communication", "Technical Leadership"], "Designs high-level technical solutions that meet business requirements."),
    ("Site Reliability Engineer", ["Monitoring", "Incident Response", "Automation", "Cloud Infrastructure", "Scripting"], "Ensures systems are reliable, scalable, and performant in production."),
    ("Business Intelligence Analyst", ["SQL", "Power BI", "Data Modeling", "Dashboards", "Business Acumen"], "Turns raw data into actionable business insights through reporting."),
    ("Financial Analyst", ["Financial Modeling", "Excel", "Forecasting", "Accounting Principles", "Reporting"], "Analyzes financial data to guide investment and business decisions."),
    ("Marketing Analyst", ["Data Analysis", "SEO", "Campaign Analysis", "Excel", "Communication"], "Analyzes marketing data to optimize campaigns and strategy."),
    ("Digital Marketing Specialist", ["SEO", "Social Media Marketing", "Content Strategy", "Analytics", "Campaign Management"], "Plans and executes digital marketing campaigns across channels."),
    ("HR Analyst", ["Data Analysis", "HR Systems", "Reporting", "Communication", "Excel"], "Uses data to support HR decisions like hiring and retention."),
    ("Operations Analyst", ["Process Improvement", "Data Analysis", "Excel", "Project Coordination", "Problem Solving"], "Analyzes and improves organizational operations and workflows."),
    ("Supply Chain Analyst", ["Logistics", "Forecasting", "Data Analysis", "Inventory Management", "Excel"], "Optimizes supply chain processes using data-driven analysis."),
    ("Actuary", ["Statistics", "Probability", "Risk Assessment", "Mathematics", "Financial Modeling"], "Assesses financial risk using mathematics and statistics, mainly in insurance."),
    ("Statistician", ["Statistics", "R", "Data Analysis", "Research Design", "Mathematics"], "Applies statistical methods to collect, analyze, and interpret data."),
    ("Bioinformatics Analyst", ["Biology", "Python", "Statistics", "Genomics Data", "Data Analysis"], "Applies computational methods to biological and genomic data."),
    ("Robotics Engineer", ["Control Systems", "Programming", "Mechanical Design", "Sensors", "Mathematics"], "Designs and builds robotic systems and automation."),
    ("Embedded Systems Engineer", ["C/C++", "Microcontrollers", "Hardware Interfacing", "RTOS", "Debugging"], "Develops software for embedded hardware devices."),
    ("Blockchain Developer", ["Solidity", "Smart Contracts", "Cryptography", "Distributed Systems", "Web3"], "Builds decentralized applications and blockchain-based systems."),
    ("Technical Support Engineer", ["Troubleshooting", "Customer Communication", "Scripting", "Documentation", "Product Knowledge"], "Provides technical assistance for software or hardware products."),
    ("Sales Engineer", ["Technical Knowledge", "Communication", "Product Demos", "Customer Relationship Management", "Negotiation"], "Combines technical expertise with sales to support customer solutions."),
    ("Management Consultant", ["Problem Solving", "Communication", "Data Analysis", "Strategy", "Presentation Skills"], "Advises organizations on improving performance and solving business problems."),
    ("Research Assistant", ["Research Methods", "Data Collection", "Literature Review", "Academic Writing", "Analysis"], "Supports academic or industry research projects."),
    ("Teacher / Tutor", ["Subject Expertise", "Communication", "Lesson Planning", "Patience", "Assessment Design"], "Educates students in academic subjects at various levels."),
    ("Content Strategist", ["Content Planning", "SEO", "Writing", "Analytics", "Editorial Judgment"], "Plans and oversees content creation aligned with business goals."),
    ("Graphic Designer", ["Adobe Creative Suite", "Typography", "Visual Composition", "Branding", "Creativity"], "Creates visual content for branding, marketing, and communication."),
    ("Compliance Officer", ["Regulatory Knowledge", "Risk Assessment", "Attention to Detail", "Reporting", "Communication"], "Ensures an organization follows laws, regulations, and internal policies."),
]

career_profiles = pd.DataFrame([
    {"career_id": i + 1, "title": title, "required_skills": skills, "description": desc}
    for i, (title, skills, desc) in enumerate(CAREERS)
])
career_profiles.to_csv(OUT_DIR / "career_profiles.csv", index=False)
print(f"{len(career_profiles)} career profiles -> {OUT_DIR / 'career_profiles.csv'}")

# --- Training pairs: positive (career's own skills <-> its title) and
# negative (career's skills <-> a different, shuffled title), mirroring
# §3.3's shape for InputExample + CosineSimilarityLoss fine-tuning.
rng = np.random.default_rng(7)

positive_pairs = career_profiles[["required_skills", "title"]].rename(columns={"title": "job_title"})
positive_pairs["label"] = 1.0

shuffled_titles = career_profiles["title"].sample(frac=1, random_state=7).reset_index(drop=True)
# guard against any row landing on its own title after shuffling
same_mask = shuffled_titles.values == career_profiles["title"].values
if same_mask.any():
    shuffled_titles = shuffled_titles.sample(frac=1, random_state=13).reset_index(drop=True)

negative_pairs = pd.DataFrame({
    "required_skills": career_profiles["required_skills"],
    "job_title": shuffled_titles,
    "label": 0.0,
})

pairs = pd.concat([positive_pairs, negative_pairs]).sample(frac=1, random_state=7).reset_index(drop=True)
pairs.to_csv(OUT_DIR / "training_pairs.csv", index=False)

print(f"{len(pairs)} training pairs -> {OUT_DIR / 'training_pairs.csv'}")
print(pairs["label"].value_counts())
