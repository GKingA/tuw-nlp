import os
from urllib.request import urlretrieve
from zipfile import ZipFile


def download_alto():
    file_path = os.path.expanduser("~/tuw_nlp_resources/alto-2.3.6-SNAPSHOT-all.jar")
    user_path = os.path.expanduser("~/tuw_nlp_resources")
    if not os.path.isfile(file_path):
        if not os.path.exists(user_path):
            os.makedirs(user_path)
            urlretrieve(
                "http://sandbox.hlt.bme.hu/~adaamko/alto-2.3.6-SNAPSHOT-all.jar",
                file_path,
            )


def download_definitions():
    file_path = os.path.expanduser("~/tuw_nlp_resources/definitions/definitions.zip")
    user_path = os.path.expanduser("~/tuw_nlp_resources/definitions")
    if not os.path.isfile(file_path):
        if not os.path.exists(user_path):
            os.makedirs(user_path)
            urlretrieve("http://sandbox.hlt.bme.hu/~adaamko/definitions.zip", file_path)

            with ZipFile(file_path, "r") as zf:
                zf.extractall(os.path.expanduser("~/tuw_nlp_resources/"))


def download_conceptnet():
    from conceptnet_lite.db import prepare_db
    basepath = os.path.expanduser("~/tuw_nlp_resources")
    if not os.path.exists(os.path.join(basepath, "conceptnet.db")):
        prepare_db(os.path.join(basepath, "conceptnet.db"))
