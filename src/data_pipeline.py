"""
data_pipeline.py
================
Module 1: Data Generation Pipeline
Generates realistic synthetic datasets:
  - users.csv  : 75 user profiles
  - feedback.csv: 500+ interaction rows

Run:
    python src/data_pipeline.py
"""

import os
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ── Constants ──────────────────────────────────────────────────────────────────
VALID_MBTI = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP",
]

# MBTI types weighted by job category for realism
ROLE_MBTI_MAP = {
    "analytical": ["INTJ", "INTP", "ISTJ", "ISTP", "ENTJ"],
    "creative":   ["INFP", "ENFP", "ISFP", "INFJ", "ENFJ"],
    "leadership": ["ENTJ", "ENFJ", "ESTJ", "ENTP", "ENFP"],
    "social":     ["ESFJ", "ESFP", "ENFJ", "ENFP", "ESTJ"],
}

CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
    "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Surat",
    "Lucknow", "Kanpur", "Nagpur", "Visakhapatnam", "Bhopal",
]

INDUSTRIES = [
    "Technology", "Healthcare", "Finance", "Education",
    "Marketing", "Design", "Data Science", "Consulting",
    "E-commerce", "Media & Entertainment",
]

GENDERS = ["Male", "Female", "Non-binary"]

# ── Name pools ──────────────────────────────────────────────────────────────────
FIRST_NAMES = [
    "Aarav", "Aisha", "Rohan", "Priya", "Kiran", "Neha", "Arjun", "Divya",
    "Vikram", "Ananya", "Rahul", "Shreya", "Amit", "Kavya", "Siddharth",
    "Pooja", "Rajesh", "Meera", "Suresh", "Lakshmi", "Nikhil", "Anjali",
    "Varun", "Sneha", "Ravi", "Deepa", "Manish", "Ritika", "Harish", "Swati",
    "Aditya", "Nandini", "Kunal", "Simran", "Tarun", "Bhavana", "Vijay", "Rekha",
    "Alok", "Tanvi", "Gaurav", "Usha", "Pranav", "Shweta", "Deepak", "Monika",
    "Sahil", "Nisha", "Raj", "Isha", "Abhinav", "Shalini", "Hemant", "Ruchika",
    "Ashish", "Vandana", "Tushar", "Madhuri", "Vivek", "Padma", "Sandeep",
    "Geeta", "Pankaj", "Sulekha", "Naveen", "Kavitha", "Sumit", "Charulata",
    "Girish", "Shravani", "Manoj", "Sujata", "Chirag", "Bhumi",
]

LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Gupta", "Kumar", "Singh", "Mehta", "Joshi",
    "Kapoor", "Reddy", "Nair", "Rao", "Pillai", "Iyer", "Mishra", "Tiwari",
    "Shukla", "Saxena", "Aggarwal", "Bose", "Chatterjee", "Das", "Sen", "Roy",
    "Ghosh", "Mukherjee", "Banerjee", "Chakraborty", "Trivedi", "Pandey",
]

# ── Professional Summary Templates ─────────────────────────────────────────────
PROF_TEMPLATES = {
    "Technology": [
        "{title} with {exp} years of experience in software development and cloud architecture. "
        "Proficient in {skills}, and passionate about building scalable, high-performance systems. "
        "Aspiring to lead engineering teams in product-driven organizations that prioritize innovation and technical excellence.",

        "Full-stack {title} specializing in {skills} with {exp} years of hands-on experience in agile environments. "
        "Focused on delivering clean, maintainable code and mentoring junior developers. "
        "Seeking roles that blend technical depth with cross-functional collaboration.",

        "Backend {title} with {exp} years building microservices and RESTful APIs using {skills}. "
        "Experienced in CI/CD pipelines, Docker, and Kubernetes. "
        "Goal-oriented professional aiming to architect resilient distributed systems at scale.",
    ],
    "Healthcare": [
        "Healthcare {title} with {exp} years of experience in clinical informatics and patient data management. "
        "Skilled in {skills} and electronic health record (EHR) systems. "
        "Committed to leveraging data to improve patient outcomes and healthcare delivery efficiency.",

        "Medical {title} with {exp} years in healthcare operations and quality improvement. "
        "Proficient in {skills} and regulatory compliance (HIPAA, ISO). "
        "Passionate about digital health transformation and evidence-based care models.",

        "{title} specializing in health analytics with {exp} years of experience using {skills}. "
        "Focused on reducing hospital readmission rates through predictive modeling. "
        "Seeking roles at the intersection of medicine and technology.",
    ],
    "Finance": [
        "Financial {title} with {exp} years in investment analysis and risk management. "
        "Expert in {skills} and quantitative modeling for portfolio optimization. "
        "Goal-driven professional targeting senior analyst roles in asset management or fintech.",

        "{title} with {exp} years of experience in corporate finance, budgeting, and forecasting. "
        "Skilled in {skills} and financial statement analysis. "
        "Passionate about driving strategic decisions through data-backed financial insights.",

        "Chartered {title} with {exp} years in tax planning and compliance. "
        "Proficient in {skills} and international tax regulations. "
        "Aiming to join a Big-4 firm or lead a finance team in a fast-growing startup.",
    ],
    "Education": [
        "Experienced {title} with {exp} years designing curriculum and delivering engaging learning experiences. "
        "Skilled in {skills} and EdTech platforms like Moodle and Canvas. "
        "Passionate about student-centered pedagogy and lifelong learning.",

        "{title} with {exp} years in corporate training and instructional design. "
        "Proficient in {skills} and learning management systems. "
        "Focused on building scalable training programs that align with organizational goals.",

        "Academic {title} with {exp} years of research experience in educational psychology and {skills}. "
        "Published author with a focus on inclusive education and adaptive learning technologies. "
        "Seeking academic or EdTech leadership positions.",
    ],
    "Marketing": [
        "Digital {title} with {exp} years driving brand growth through SEO, content strategy, and paid media. "
        "Expert in {skills} and marketing analytics platforms. "
        "Results-oriented professional with a track record of increasing organic traffic by 200% YoY.",

        "{title} with {exp} years in B2B marketing and demand generation. "
        "Skilled in {skills}, HubSpot, and account-based marketing (ABM). "
        "Passionate about aligning sales and marketing for maximum revenue impact.",

        "Growth {title} with {exp} years specializing in product-led growth and user acquisition funnels. "
        "Proficient in {skills} and A/B testing frameworks. "
        "Aspiring to lead marketing strategy at a Series B+ startup.",
    ],
    "Design": [
        "UX/UI {title} with {exp} years crafting intuitive digital experiences for web and mobile platforms. "
        "Expert in {skills} and design systems, with a strong portfolio across fintech and healthcare. "
        "Passionate about human-centered design and accessibility.",

        "Product {title} with {exp} years bridging design and engineering through design thinking methodologies. "
        "Skilled in {skills} and rapid prototyping. "
        "Seeking to join product teams that value creativity and user research.",

        "Brand {title} with {exp} years in visual identity and communication design. "
        "Proficient in {skills} and motion graphics. "
        "Focused on creating cohesive brand narratives that resonate with target audiences.",
    ],
    "Data Science": [
        "Data Scientist with {exp} years applying machine learning and statistical modeling to business problems. "
        "Expert in {skills} and deep learning frameworks. "
        "Passionate about building explainable AI systems that drive real-world impact.",

        "{title} with {exp} years in NLP, computer vision, and recommendation systems. "
        "Proficient in {skills} and cloud ML platforms (AWS SageMaker, GCP Vertex AI). "
        "Goal-oriented professional targeting senior ML engineer roles in product companies.",

        "ML {title} with {exp} years designing end-to-end data pipelines and predictive models. "
        "Skilled in {skills} and MLOps practices for model deployment and monitoring. "
        "Aspiring to lead AI research teams focused on ethical and responsible AI.",
    ],
    "Consulting": [
        "Management {title} with {exp} years advising Fortune 500 clients on digital transformation. "
        "Expert in {skills} and organizational change management. "
        "Passionate about solving complex business problems with data-driven strategy.",

        "{title} with {exp} years in process optimization and operational excellence. "
        "Skilled in {skills} and Six Sigma methodologies. "
        "Focused on delivering measurable ROI through structured problem-solving frameworks.",

        "Strategy {title} with {exp} years in market entry, M&A due diligence, and competitive analysis. "
        "Proficient in {skills} and executive presentation skills. "
        "Seeking senior advisory roles in top-tier consulting firms.",
    ],
    "E-commerce": [
        "{title} with {exp} years managing product catalogues, pricing strategy, and customer experience on e-commerce platforms. "
        "Expert in {skills} and marketplace analytics (Amazon Seller Central, Flipkart). "
        "Driven by a passion for scaling D2C brands through data-informed decisions.",

        "E-commerce {title} with {exp} years in supply chain optimization and last-mile delivery management. "
        "Skilled in {skills} and logistics platforms. "
        "Committed to improving operational efficiency and customer satisfaction.",

        "Growth {title} with {exp} years in conversion rate optimization, email marketing, and retention strategies. "
        "Proficient in {skills} and Shopify Plus. "
        "Aiming to scale e-commerce revenues to INR 100 Cr for a leading D2C brand.",
    ],
    "Media & Entertainment": [
        "Content {title} with {exp} years creating engaging digital media for YouTube, OTT, and social platforms. "
        "Skilled in {skills} and SEO-driven content strategy. "
        "Passionate about storytelling and building loyal audiences.",

        "{title} with {exp} years in film production, scriptwriting, and post-production workflows. "
        "Proficient in {skills} and digital distribution strategies. "
        "Seeking creative lead roles in OTT platforms or digital studios.",

        "Media {title} with {exp} years in brand partnerships, influencer marketing, and audience analytics. "
        "Expert in {skills} and monetization strategies for creator economy platforms. "
        "Focused on building authentic content ecosystems.",
    ],
}

SKILLS_MAP = {
    "Technology":         ["Python", "JavaScript", "React", "Node.js", "AWS", "Docker", "Kubernetes", "SQL", "Java", "Go"],
    "Healthcare":         ["EHR Systems", "Python", "SQL", "HIPAA Compliance", "Power BI", "Clinical Analytics", "HL7 FHIR"],
    "Finance":            ["Excel", "Python", "SQL", "Bloomberg Terminal", "VBA", "Financial Modeling", "SAP", "Tableau"],
    "Education":          ["Curriculum Design", "Moodle", "Google Classroom", "Instructional Design", "Python", "SPSS"],
    "Marketing":          ["Google Analytics", "SEO", "HubSpot", "Meta Ads", "Salesforce", "Tableau", "Python", "Mailchimp"],
    "Design":             ["Figma", "Adobe XD", "Sketch", "Illustrator", "Photoshop", "InVision", "Zeplin", "Principle"],
    "Data Science":       ["Python", "R", "TensorFlow", "PyTorch", "SQL", "Spark", "Scikit-learn", "Tableau", "Jupyter"],
    "Consulting":         ["PowerPoint", "Excel", "Python", "JIRA", "Tableau", "Power BI", "Process Mapping", "SQL"],
    "E-commerce":         ["Shopify", "Python", "Google Analytics", "SQL", "Excel", "Meta Ads", "Klaviyo", "Tableau"],
    "Media & Entertainment": ["Adobe Premiere", "Final Cut Pro", "After Effects", "DaVinci Resolve", "YouTube Studio", "Canva"],
}

TITLES_MAP = {
    "Technology":         ["Software Engineer", "Backend Developer", "Full-Stack Developer", "DevOps Engineer", "Cloud Architect"],
    "Healthcare":         ["Health Informatics Specialist", "Clinical Data Analyst", "Healthcare Consultant", "Medical Officer"],
    "Finance":            ["Financial Analyst", "Investment Banker", "Risk Manager", "Chartered Accountant", "CFO"],
    "Education":          ["Educator", "Curriculum Designer", "Training Specialist", "Academic Researcher", "Instructional Designer"],
    "Marketing":          ["Digital Marketer", "Growth Hacker", "Content Strategist", "Brand Manager", "Marketing Analyst"],
    "Design":             ["UX Designer", "Product Designer", "UI Engineer", "Brand Designer", "Visual Designer"],
    "Data Science":       ["Data Scientist", "ML Engineer", "AI Researcher", "Data Analyst", "NLP Engineer"],
    "Consulting":         ["Management Consultant", "Strategy Analyst", "Business Consultant", "Operations Consultant"],
    "E-commerce":         ["E-commerce Manager", "Marketplace Analyst", "D2C Growth Lead", "Operations Manager"],
    "Media & Entertainment": ["Content Creator", "Video Producer", "Media Analyst", "Brand Partnerships Manager"],
}

ROLE_CATEGORY = {
    "Technology":         "analytical",
    "Healthcare":         "analytical",
    "Finance":            "analytical",
    "Education":          "social",
    "Marketing":          "creative",
    "Design":             "creative",
    "Data Science":       "analytical",
    "Consulting":         "leadership",
    "E-commerce":         "leadership",
    "Media & Entertainment": "creative",
}

# ── About Me Templates ──────────────────────────────────────────────────────────
ABOUT_TEMPLATES_BY_MBTI_GROUP = {
    "analytical": [
        "I thrive in structured environments where logic and precision drive outcomes. "
        "I enjoy deep-diving into complex problems, reading about emerging technologies, and building side projects on weekends. "
        "I value intellectual honesty, continuous learning, and meaningful conversations over small talk.",

        "Naturally curious and detail-oriented, I approach challenges with a systematic mindset. "
        "Outside work, I enjoy solving competitive programming problems, exploring philosophy, and hiking in quiet places. "
        "I believe the best teams are built on trust, transparency, and mutual respect.",

        "I am an introvert who does my best thinking in focused, distraction-free environments. "
        "I enjoy reading non-fiction, writing technical blogs, and contributing to open-source projects. "
        "I value colleagues who are direct, dependable, and always hungry to learn.",

        "I find satisfaction in optimizing systems and uncovering patterns in data. "
        "In my free time, I enjoy chess, documentary films, and cooking elaborate meals as a creative outlet. "
        "I work best in collaborative but quiet teams where quality is prioritized over speed.",
    ],
    "creative": [
        "I am a creative soul who finds inspiration in art, travel, and human stories. "
        "I believe design is not just aesthetics but a language that connects people. "
        "I love brainstorming sessions, cross-disciplinary collaboration, and pushing boundaries of what is possible.",

        "Empathetic and imaginative, I bring fresh perspectives to every project I touch. "
        "I enjoy journaling, attending indie music gigs, and experimenting with photography. "
        "My best work happens when I have the freedom to explore ideas without rigid constraints.",

        "I am driven by the belief that great ideas change the world. "
        "I love collaborating with diverse teams, mentoring aspiring designers, and volunteering for community education initiatives. "
        "I thrive in environments that value creativity, flexibility, and human-centered thinking.",

        "Storytelling is at the heart of everything I do—whether writing a product brief or designing a campaign. "
        "I am passionate about social impact, sustainable design, and inclusive communication. "
        "Outside work, I enjoy pottery, fiction writing, and weekend trekking.",
    ],
    "leadership": [
        "I am naturally energized by leading teams and turning ambitious visions into measurable results. "
        "I love public speaking, strategic planning, and building networks across industries. "
        "I believe great leaders grow other leaders, and I actively mentor young professionals in my field.",

        "Results-focused and people-first—that is my leadership philosophy. "
        "I enjoy networking events, competitive sports, and reading business biographies. "
        "I work best in fast-paced environments where decisions need to be made quickly and boldly.",

        "I am an extrovert who draws energy from collaborative, high-stakes projects. "
        "I enjoy organizing team offsites, facilitating workshops, and coaching colleagues through career transitions. "
        "I value accountability, ambition, and a positive team culture above all else.",

        "Strategic thinking and relationship-building are my superpowers. "
        "I enjoy attending industry conferences, debating policy and economics, and playing team sports. "
        "I thrive in roles where I can influence decisions and shape organizational culture.",
    ],
    "social": [
        "I am a warm, supportive person who genuinely cares about the well-being of my colleagues and community. "
        "I enjoy organizing study groups, volunteering at local schools, and cooking meals for friends and family. "
        "I work best in harmonious, collaborative teams where everyone feels valued and heard.",

        "People are my greatest passion—helping them grow, learn, and reach their potential. "
        "I love hosting team socials, facilitating icebreaker activities, and creating inclusive spaces. "
        "I value kindness, empathy, and a sense of humor in the workplace.",

        "I am a dedicated community builder who believes meaningful work starts with meaningful relationships. "
        "Outside work, I enjoy theatre, reading self-help books, and participating in cultural events. "
        "I thrive in environments that prioritize collaboration, diversity, and open communication.",

        "Optimistic and team-oriented, I bring enthusiasm and care to everything I do. "
        "I enjoy mentoring interns, leading team retrospectives, and celebrating small wins together. "
        "I believe a positive work culture is just as important as any technical skill.",
    ],
}


def pick_skills(industry: str, n: int = 4) -> str:
    pool = SKILLS_MAP.get(industry, ["Excel", "Python", "SQL"])
    return ", ".join(random.sample(pool, min(n, len(pool))))


def generate_users(n: int = 75) -> pd.DataFrame:
    users = []
    used_names = set()

    for i in range(n):
        user_id = f"U{str(i + 1).zfill(3)}"
        industry = random.choice(INDUSTRIES)
        role_cat = ROLE_CATEGORY[industry]
        mbti = random.choice(ROLE_MBTI_MAP[role_cat])
        title = random.choice(TITLES_MAP[industry])
        exp = random.randint(1, 12)
        skills_str = pick_skills(industry, n=random.randint(3, 5))

        # Generate unique name
        attempts = 0
        while True:
            fname = random.choice(FIRST_NAMES)
            lname = random.choice(LAST_NAMES)
            full_name = f"{fname} {lname}"
            if full_name not in used_names or attempts > 50:
                used_names.add(full_name)
                break
            attempts += 1

        age = random.randint(22, 45)
        gender = random.choice(GENDERS)
        city = random.choice(CITIES)

        # Professional summary
        tmpl = random.choice(PROF_TEMPLATES[industry])
        prof_summary = tmpl.format(title=title, exp=exp, skills=skills_str)

        # About me
        about_me = random.choice(ABOUT_TEMPLATES_BY_MBTI_GROUP[role_cat])

        users.append({
            "user_id": user_id,
            "name": full_name,
            "age": age,
            "gender": gender,
            "location": city,
            "industry": industry,
            "job_title": title,
            "mbti_type": mbti,
            "skills": skills_str,
            "professional_summary": prof_summary,
            "about_me": about_me,
        })

    return pd.DataFrame(users)


def generate_feedback(users_df: pd.DataFrame, interactions_per_user: int = 7) -> pd.DataFrame:
    """
    Simulate realistic feedback: users tend to accept matches
    in similar industries (~70% accept) and random others (~40% accept).
    """
    records = []
    user_ids = users_df["user_id"].tolist()
    industry_map = dict(zip(users_df["user_id"], users_df["industry"]))

    base_date = datetime(2025, 1, 10)

    for uid in user_ids:
        matched_pool = [x for x in user_ids if x != uid]
        num_interactions = random.randint(interactions_per_user - 2, interactions_per_user + 3)
        sampled = random.sample(matched_pool, min(num_interactions, len(matched_pool)))

        for mid in sampled:
            same_industry = industry_map[uid] == industry_map[mid]
            # Realistic acceptance probability
            accept_prob = 0.68 if same_industry else 0.42
            action = 1 if random.random() < accept_prob else 0
            days_offset = random.randint(0, 180)
            ts = (base_date + timedelta(days=days_offset)).strftime("%Y-%m-%d")
            records.append({
                "user_id": uid,
                "matched_user_id": mid,
                "action": action,
                "timestamp": ts,
            })

    df = pd.DataFrame(records)
    # Remove duplicates (same pair)
    df = df.drop_duplicates(subset=["user_id", "matched_user_id"])
    return df.reset_index(drop=True)


def run_pipeline():
    os.makedirs("data", exist_ok=True)

    print("Generating user profiles...")
    users_df = generate_users(n=75)
    users_df.to_csv("data/users.csv", index=False)
    print(f"  ✓ data/users.csv created — {len(users_df)} users")

    print("Generating feedback interactions...")
    feedback_df = generate_feedback(users_df, interactions_per_user=7)
    feedback_df.to_csv("data/feedback.csv", index=False)
    print(f"  ✓ data/feedback.csv created — {len(feedback_df)} interactions")

    print("\nSample users:")
    print(users_df[["user_id", "name", "industry", "mbti_type", "location"]].head(5).to_string(index=False))

    print("\nFeedback stats:")
    print(f"  Total rows      : {len(feedback_df)}")
    print(f"  Acceptance rate : {feedback_df['action'].mean():.1%}")
    return users_df, feedback_df


if __name__ == "__main__":
    run_pipeline()
