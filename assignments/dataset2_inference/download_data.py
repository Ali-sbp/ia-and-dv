"""Download and verify the two Dataset 2 course files without adding them to Git."""

import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import urlencode
from urllib.request import urlopen

from analysis import CARDS, DATA, PREDICTIONS

COURSE = "https://disk.yandex.ru/d/waiZmiOzun8EZA"
REMOTE = "/3.6_Пр_Интеграция_ML/BI-трек (практика)/Qlik/Data/3/Parquet/Gen"
SHA256 = {
    PREDICTIONS: "eed90d79ffab0fd485f97cbe899c351aca62f9293ba4ef1c40199eda377baed2",
    CARDS: "00c7e45e5fdfd361fc631bdd2a621940f0df3804d16a3ebd750f719545658c64",
}


def digest(path: Path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    DATA.mkdir(exist_ok=True, parents=True)
    needed = [name for name, expected in SHA256.items()
              if not (DATA / name).is_file() or digest(DATA / name) != expected]
    if needed:
        params = urlencode({"public_key": COURSE, "path": REMOTE, "limit": 100})
        with urlopen(f"https://cloud-api.yandex.net/v1/disk/public/resources?{params}", timeout=30) as r:
            files = {item["name"]: item for item in json.load(r)["_embedded"]["items"]}
        for name in needed:
            target = DATA / name
            temp = target.with_suffix(".part")
            try:
                with urlopen(files[name]["file"], timeout=60) as source, temp.open("wb") as out:
                    shutil.copyfileobj(source, out, length=1024 * 1024)
                if digest(temp) != SHA256[name]:
                    raise ValueError(f"Контрольная сумма файла {name} не совпала.")
                temp.replace(target)
            finally:
                temp.unlink(missing_ok=True)
    for name in SHA256:
        print(f"Файл проверен: {DATA / name}")


if __name__ == "__main__":
    main()
