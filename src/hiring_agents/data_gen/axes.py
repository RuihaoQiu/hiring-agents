from __future__ import annotations

ROLE_FAMILIES: tuple[str, ...] = (
    "backend",
    "frontend",
    "fullstack",
    "data_scientist",
    "ml_engineer",
    "devops",
    "mobile",
    "product_manager",
    "designer",
)

SENIORITY_YOE_RANGE: dict[str, tuple[int, int]] = {
    "junior": (1, 3),
    "mid": (3, 6),
    "senior": (6, 10),
    "staff": (10, 18),
}

LOCATIONS: tuple[str, ...] = (
    "Berlin, Germany",
    "Munich, Germany",
    "Hamburg, Germany",
    "Amsterdam, Netherlands",
    "London, United Kingdom",
    "Paris, France",
    "Barcelona, Spain",
    "Madrid, Spain",
    "Lisbon, Portugal",
    "Stockholm, Sweden",
    "Copenhagen, Denmark",
    "Zurich, Switzerland",
    "Dublin, Ireland",
    "Remote EU",
)

TECH_STACKS: dict[str, tuple[tuple[str, ...], ...]] = {
    "backend": (
        ("Python", "Django", "Postgres", "Redis"),
        ("Python", "FastAPI", "Postgres", "Kafka"),
        ("Java", "Spring", "Kafka", "MySQL"),
        ("Go", "gRPC", "Kubernetes", "Postgres"),
        ("Node.js", "Express", "MongoDB", "Redis"),
        ("Ruby", "Rails", "Postgres", "Sidekiq"),
        ("Rust", "Tokio", "Postgres"),
        ("C#", ".NET", "SQL Server", "Azure"),
    ),
    "frontend": (
        ("React", "TypeScript", "Next.js", "Tailwind"),
        ("Vue", "Nuxt", "TypeScript", "Vite"),
        ("Angular", "TypeScript", "RxJS"),
        ("Svelte", "SvelteKit", "TypeScript"),
        ("React", "Redux", "Webpack", "JavaScript"),
    ),
    "fullstack": (
        ("Python", "Django", "React", "Postgres"),
        ("Node.js", "Express", "React", "MongoDB"),
        ("TypeScript", "Next.js", "tRPC", "Postgres"),
        ("Ruby", "Rails", "React", "Postgres"),
    ),
    "data_scientist": (
        ("Python", "pandas", "scikit-learn", "SQL"),
        ("Python", "PyTorch", "Jupyter", "SQL"),
        ("R", "tidyverse", "SQL", "Tableau"),
        ("Python", "XGBoost", "MLflow", "Spark"),
    ),
    "ml_engineer": (
        ("Python", "PyTorch", "AWS SageMaker", "Docker"),
        ("Python", "TensorFlow", "Kubeflow", "GCP"),
        ("Python", "Ray", "MLflow", "Kubernetes"),
        ("Python", "Hugging Face", "ONNX", "Triton"),
    ),
    "devops": (
        ("Kubernetes", "Terraform", "AWS", "Python"),
        ("Docker", "Ansible", "GCP", "Bash"),
        ("Kubernetes", "Helm", "ArgoCD", "Prometheus"),
        ("Terraform", "Azure", "GitHub Actions", "PowerShell"),
    ),
    "mobile": (
        ("Swift", "iOS", "Combine", "CoreData"),
        ("Kotlin", "Android", "Jetpack Compose", "Coroutines"),
        ("React Native", "TypeScript", "Expo"),
        ("Flutter", "Dart", "Firebase"),
    ),
    "product_manager": (
        ("SQL", "Amplitude", "Figma", "Jira"),
        ("SQL", "Mixpanel", "Notion", "Linear"),
        ("SQL", "Looker", "Figma"),
    ),
    "designer": (
        ("Figma", "Design Systems", "Prototyping"),
        ("Figma", "Sketch", "User Research"),
        ("Figma", "Framer", "Motion"),
    ),
}

DOMAINS: tuple[str, ...] = (
    "fintech",
    "healthcare",
    "e-commerce",
    "gaming",
    "enterprise_saas",
    "consumer",
    "logistics",
    "adtech",
    "edtech",
    "cybersecurity",
)

REGISTERS: tuple[str, ...] = (
    "terse bullet points",
    "narrative paragraphs",
    "mixed bullets and prose",
    "dense single paragraph",
)

QUALITY_LEVELS: tuple[str, ...] = (
    "polished",
    "slightly informal",
    "a few typos and inconsistent capitalization",
    "unpolished with abbreviations",
)

TRAIT_NONE: str = "none"

MESSY_TRAITS: tuple[str, ...] = (
    "career gap of ~18 months",
    "recently switched role families",
    "transitioning from IC to manager",
    "works at a tiny startup with inflated title",
    "location is ambiguous (e.g. 'Berlin / remote')",
    "has an uncommon niche skill combination",
)
