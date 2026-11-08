"""Fetch the original Dataset 1 export from the course's shared Yandex Disk."""

import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import urlencode
from urllib.request import urlopen

from analysis import HERE, NAME

COURSE = "https://disk.yandex.ru/d/waiZmiOzun8EZA"
REMOTE = f"/1.5 Пр Декомпозиция/Dataset/{NAME}"
SHA256 = "c3a4f35d96f9d67137e7462576d467c116a3c370c81100082afb18818e7c9657"


def digest(path: Path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    target = HERE / "data" / NAME
    if target.is_file() and digest(target) == SHA256:
        print(f"Файл уже проверен: {target}")
        return
    target.parent.mkdir(exist_ok=True)
    params = urlencode({"public_key": COURSE, "path": REMOTE})
    with urlopen(f"https://cloud-api.yandex.net/v1/disk/public/resources?{params}", timeout=30) as response:
        metadata = json.load(response)
    temp = target.with_suffix(".part")
    try:
        with urlopen(metadata["file"], timeout=60) as response, temp.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if digest(temp) != SHA256:
            raise ValueError("Контрольная сумма не совпала с выгрузкой практики 1.5.")
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    print(f"Файл проверен: {target}")


if __name__ == "__main__":
    main()
