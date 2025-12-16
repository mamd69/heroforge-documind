#!/usr/bin/env python3
"""
Skill Packager - Creates a distributable .skill file of a skill folder

Usage:
<<<<<<< HEAD
    python package_skill.py <path/to/skill-folder> [output-directory]

Example:
    python package_skill.py skills/public/my-skill
    python package_skill.py skills/public/my-skill ./dist
=======
    python utils/package_skill.py <path/to/skill-folder> [output-directory]

Example:
    python utils/package_skill.py skills/public/my-skill
    python utils/package_skill.py skills/public/my-skill ./dist
>>>>>>> c4b60ab0f0fdb6fc8f492169c352a50d42140bc3
"""

import sys
import zipfile
from pathlib import Path
<<<<<<< HEAD


def validate_skill(skill_path):
    """
    Validate a skill folder meets basic requirements.

    Args:
        skill_path: Path to the skill folder

    Returns:
        Tuple of (is_valid, message)
    """
    skill_path = Path(skill_path)

    # Check SKILL.md exists
    skill_md = skill_path / "SKILL.md"
    if not skill_md.exists():
        return False, "SKILL.md not found"

    # Read and validate frontmatter
    content = skill_md.read_text()
    if not content.startswith("---"):
        return False, "SKILL.md must start with YAML frontmatter (---)"

    # Check for required fields
    if "name:" not in content:
        return False, "SKILL.md frontmatter must contain 'name' field"
    if "description:" not in content:
        return False, "SKILL.md frontmatter must contain 'description' field"

    # Check description is not a TODO
    if "[TODO:" in content.split("---")[1]:
        return False, "SKILL.md frontmatter contains TODO placeholders - please complete them"

    return True, "Skill validation passed"
=======
from quick_validate import validate_skill
>>>>>>> c4b60ab0f0fdb6fc8f492169c352a50d42140bc3


def package_skill(skill_path, output_dir=None):
    """
    Package a skill folder into a .skill file.

    Args:
        skill_path: Path to the skill folder
        output_dir: Optional output directory for the .skill file (defaults to current directory)

    Returns:
        Path to the created .skill file, or None if error
    """
    skill_path = Path(skill_path).resolve()

    # Validate skill folder exists
    if not skill_path.exists():
<<<<<<< HEAD
        print(f"Error: Skill folder not found: {skill_path}")
        return None

    if not skill_path.is_dir():
        print(f"Error: Path is not a directory: {skill_path}")
=======
        print(f"❌ Error: Skill folder not found: {skill_path}")
        return None

    if not skill_path.is_dir():
        print(f"❌ Error: Path is not a directory: {skill_path}")
>>>>>>> c4b60ab0f0fdb6fc8f492169c352a50d42140bc3
        return None

    # Validate SKILL.md exists
    skill_md = skill_path / "SKILL.md"
    if not skill_md.exists():
<<<<<<< HEAD
        print(f"Error: SKILL.md not found in {skill_path}")
        return None

    # Run validation before packaging
    print("Validating skill...")
    valid, message = validate_skill(skill_path)
    if not valid:
        print(f"Validation failed: {message}")
        print("   Please fix the validation errors before packaging.")
        return None
    print(f"{message}\n")
=======
        print(f"❌ Error: SKILL.md not found in {skill_path}")
        return None

    # Run validation before packaging
    print("🔍 Validating skill...")
    valid, message = validate_skill(skill_path)
    if not valid:
        print(f"❌ Validation failed: {message}")
        print("   Please fix the validation errors before packaging.")
        return None
    print(f"✅ {message}\n")
>>>>>>> c4b60ab0f0fdb6fc8f492169c352a50d42140bc3

    # Determine output location
    skill_name = skill_path.name
    if output_dir:
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path.cwd()

    skill_filename = output_path / f"{skill_name}.skill"

    # Create the .skill file (zip format)
    try:
        with zipfile.ZipFile(skill_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Walk through the skill directory
            for file_path in skill_path.rglob('*'):
                if file_path.is_file():
                    # Calculate the relative path within the zip
                    arcname = file_path.relative_to(skill_path.parent)
                    zipf.write(file_path, arcname)
                    print(f"  Added: {arcname}")

<<<<<<< HEAD
        print(f"\nSuccessfully packaged skill to: {skill_filename}")
        return skill_filename

    except Exception as e:
        print(f"Error creating .skill file: {e}")
=======
        print(f"\n✅ Successfully packaged skill to: {skill_filename}")
        return skill_filename

    except Exception as e:
        print(f"❌ Error creating .skill file: {e}")
>>>>>>> c4b60ab0f0fdb6fc8f492169c352a50d42140bc3
        return None


def main():
    if len(sys.argv) < 2:
<<<<<<< HEAD
        print("Usage: python package_skill.py <path/to/skill-folder> [output-directory]")
        print("\nExample:")
        print("  python package_skill.py skills/public/my-skill")
        print("  python package_skill.py skills/public/my-skill ./dist")
=======
        print("Usage: python utils/package_skill.py <path/to/skill-folder> [output-directory]")
        print("\nExample:")
        print("  python utils/package_skill.py skills/public/my-skill")
        print("  python utils/package_skill.py skills/public/my-skill ./dist")
>>>>>>> c4b60ab0f0fdb6fc8f492169c352a50d42140bc3
        sys.exit(1)

    skill_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

<<<<<<< HEAD
    print(f"Packaging skill: {skill_path}")
=======
    print(f"📦 Packaging skill: {skill_path}")
>>>>>>> c4b60ab0f0fdb6fc8f492169c352a50d42140bc3
    if output_dir:
        print(f"   Output directory: {output_dir}")
    print()

    result = package_skill(skill_path, output_dir)

    if result:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
