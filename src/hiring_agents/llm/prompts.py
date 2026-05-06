from hiring_agents.config import SENIORITY_VOCAB, SUMMARY_WORD_MAX, SUMMARY_WORD_MIN

RESUME_GENERATION_SYSTEM = """You write realistic resume text for synthetic candidates.

Rules:
- Vary register: some terse and bulleted, some narrative, some mixed.
- Vary quality: occasional typos, inconsistent capitalization, abbreviations.
- Include company names (plausible or real), dates, and specific responsibilities.
- Describe actual work, not skill name-dropping.
- NO editorializing ("passionate", "rockstar", "ninja", "synergy").
- Length: roughly 250-450 words. Do not output markdown headers.
- Output ONLY the resume text. No preamble, no commentary."""

RESUME_GENERATION_USER = """Write a resume for a candidate with these hidden attributes:

- Role family: {role_family}
- Seniority: {seniority} ({yoe_range} years of experience — pick a specific number in range)
- Primary location: {location}
- Core tech stack: {tech_stack}
- Industry domain: {domain}
- Register/style: {register}
- Quality: {quality}
- Special trait: {trait}

Make the resume internally consistent with these attributes but do NOT list them verbatim;
write prose, bullets, or mixed as a real resume would."""

STRUCTURED_EXTRACTION_SYSTEM = """You extract structured fields from resume text.
Return JSON only, matching the provided schema exactly. If a field is unknown,
use your best inference from context. total_yoe is an integer number of years."""

SUMMARY_SYSTEM = f"""You write a dense canonical summary of a candidate for retrieval.

Rules:
- {SUMMARY_WORD_MIN}-{SUMMARY_WORD_MAX} words.
- Expand abbreviations (k8s -> Kubernetes, ML -> machine learning) but keep both forms
  where both are commonly searched.
- Preserve specificity: named tools, years of experience, domains, seniority, location.
- No editorializing. No bullet points. Plain prose, one block.
- Mention: role family, seniority, years of experience, current focus, primary skills,
  notable domains, location."""

SUMMARY_USER = """Resume:
{resume_text}

Structured fields already extracted:
{structured_json}

Write the canonical summary."""

SENIORITY_INFERENCE_SYSTEM = (
    "Classify the seniority level of a candidate from their current title and work history.\n"
    f"Choose exactly one from: {', '.join(SENIORITY_VOCAB)}.\n"
    "Return only that single word, nothing else."
)

QUERY_NORMALIZATION_SYSTEM = """You normalize a recruiter's free-text search into a
structured query for candidate retrieval. Return JSON only.

- hard_filters: location_keywords (list of city/country strings), seniority (list of
  seniority levels from: junior, mid, senior, staff, principal). Only set seniority when
  the query explicitly signals a level (e.g. "senior", "junior", "lead", "staff").
  Use null for anything the query doesn't specify.
- core_skills: concrete required skills/technologies.
- nice_to_haves: bonuses, not required.
- canonical_summary: a 100-150 word dense description of the ideal candidate, written
  in the same style as candidate summaries (no editorializing, concrete specifics)."""

QUERY_NORMALIZATION_USER = """Recruiter input:
{raw_query}

Normalize it."""

RERANK_SYSTEM = """You score a candidate's fit for a structured query. Return JSON only.

- score: 1 (poor fit) to 5 (excellent fit).
- must_have_matches: one entry per item in the query's core_skills. Each entry has
  requirement (the skill), met (bool), evidence (short quote or paraphrase from the
  resume, or empty string if not met).
- gaps: short list of the most important missing requirements or concerns.
- one_line_summary: one sentence, factual, no editorializing.

Scoring rules:
- Base score on must-have coverage, seniority alignment, domain relevance, location alignment.
- If the query specifies a seniority level (senior, staff, lead, principal) and the candidate
  is clearly more junior, cap the score at 2 regardless of skill coverage.
- If the query specifies junior/mid and the candidate is overqualified, note it in gaps but
  do not penalize heavily.
- Ignore fluff."""

RERANK_USER = """Query:
{normalized_query_json}

Candidate:
{candidate_payload_json}

Score this candidate."""
