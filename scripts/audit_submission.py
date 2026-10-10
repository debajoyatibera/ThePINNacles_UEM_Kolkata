from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def check_file(relative_path: str) -> bool:
    path = ROOT / relative_path
    exists = path.exists()

    status = "PASS" if exists else "MISSING"
    print(f"[{status}] {relative_path}")

    return exists


def check_readme_contains(term: str) -> bool:
    readme = ROOT / "README.md"

    if not readme.exists():
        print("[MISSING] README.md")
        return False

    content = readme.read_text(encoding="utf-8").lower()
    found = term.lower() in content

    status = "PASS" if found else "MISSING"
    print(f"[{status}] README contains: {term}")

    return found


def main() -> None:
    print("=" * 70)
    print("CRP DIGITAL TWIN — SUBMISSION AUDIT")
    print("=" * 70)
    print()

    print("CORE FILES")
    print("-" * 70)

    check_file("README.md")
    check_file("requirements.txt")
    check_file("pyproject.toml")

    print()
    print("SUBMISSION MATERIALS")
    print("-" * 70)

    check_file("LICENSE")

    print()
    print("README REQUIREMENTS")
    print("-" * 70)

    required_sections = [
        "Team Details",
        "University of Engineering & Management",
        "Project Title",
        "Problem Statement",
        "Healthcare Use Case",
        "Technical Stack",
        "Physics-Informed",
        "Demo Video",
        "Architecture Diagram",
        "Project Presentation",
        "Open-Source License",
        "Limitations",
    ]

    for section in required_sections:
        check_readme_contains(section)

    print()
    print("PROJECT STRUCTURE")
    print("-" * 70)

    check_file("src/biotwin")
    check_file("tests")
    check_file("data")
    check_file("outputs")

    print()
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()