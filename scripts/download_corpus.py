"""
Lädt die FastAPI-Dokumentation als Korpus.

Die FastAPI-Docs liegen als Markdown auf GitHub:
    https://github.com/fastapi/fastapi/tree/master/docs/en/docs

Wir ziehen eine kuratierte Auswahl an relevanten Seiten:
    - tutorial/  (Grundlagen)
    - advanced/  (fortgeschrittene Themen)
    - deployment/ (Deployment-Konzepte)
"""
import sys
from pathlib import Path

import httpx
from tqdm import tqdm

# FastAPI docs source (raw GitHub)
BASE_URL = "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs"

# Kuratierte Liste — relevant und nicht zu groß
FILES = [
    # Tutorial
    "tutorial/index.md",
    "tutorial/first-steps.md",
    "tutorial/path-params.md",
    "tutorial/query-params.md",
    "tutorial/body.md",
    "tutorial/query-params-str-validations.md",
    "tutorial/path-params-numeric-validations.md",
    "tutorial/body-multiple-params.md",
    "tutorial/body-fields.md",
    "tutorial/body-nested-models.md",
    "tutorial/extra-models.md",
    "tutorial/response-model.md",
    "tutorial/response-status-code.md",
    "tutorial/request-files.md",
    "tutorial/request-forms.md",
    "tutorial/handling-errors.md",
    "tutorial/body-updates.md",
    "tutorial/dependencies/index.md",
    "tutorial/security/index.md",
    "tutorial/middleware.md",
    "tutorial/cors.md",
    "tutorial/sql-databases.md",
    "tutorial/bigger-applications.md",
    "tutorial/background-tasks.md",
    "tutorial/metadata.md",
    "tutorial/testing.md",
    "tutorial/debugging.md",
    # Advanced
    "advanced/index.md",
    "advanced/path-operation-advanced-configuration.md",
    "advanced/additional-status-codes.md",
    "advanced/response-directly.md",
    "advanced/custom-response.md",
    "advanced/additional-responses.md",
    "advanced/response-cookies.md",
    "advanced/response-headers.md",
    "advanced/response-change-status-code.md",
    "advanced/advanced-dependencies.md",
    "advanced/websockets.md",
    "advanced/events.md",
    "advanced/testing-websockets.md",
    "advanced/testing-dependencies.md",
    "advanced/testing-database.md",
]


def download_file(client: httpx.Client, rel_path: str, target_dir: Path) -> bool:
    """Lädt eine einzelne Datei. True bei Erfolg."""
    url = f"{BASE_URL}/{rel_path}"
    target = target_dir / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = client.get(url, timeout=20.0, follow_redirects=True)
        response.raise_for_status()
        target.write_text(response.text, encoding="utf-8")
        return True
    except Exception as e:
        print(f"  ⚠ {rel_path}: {e}", file=sys.stderr)
        return False


def main():
    target_dir = Path(__file__).parent.parent / "data" / "raw" / "fastapi-docs"
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Lade FastAPI-Dokumentation nach: {target_dir}")
    print(f"   {len(FILES)} Dateien")

    ok = 0
    fail = 0

    with httpx.Client() as client:
        for rel_path in tqdm(FILES, desc="Downloading"):
            if download_file(client, rel_path, target_dir):
                ok += 1
            else:
                fail += 1

    print(f"\n✅ Fertig: {ok} erfolgreich, {fail} fehlgeschlagen")
    print(f"   Ziel: {target_dir}")


if __name__ == "__main__":
    main()