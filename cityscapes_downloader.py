import os
import subprocess
import zipfile
import requests
import glob


class CityscapesDownloader:
    # TODO: find out whats the expected config system and use that instead of hard coding
    package_mapping = {
        'gtFine': '1',
        'gtCoarse': '2',
        'leftImg8bit': '3',
        'leftImg8bit-extra': '4',
        'camera': '8',
        'camera-extra': '9',
        'vehicle': '10',
        'vehicle-extra': '11',
        'leftImg8bit-demo': '12',
        'gtBbox-cityPersons': '28'
    }

    def __init__(self, root, login, packages):
        self.root = root
        self.login = login
        self.packages = packages

    def download(self):

        for key in self.packages:
            subprocess.check_call([
                'resources/cityscapes-download-script.sh',
                self.login['user'],
                self.login['pass'],
                self.package_mapping[key]
            ])

        # TODO: See whats happening with these requests and delete the dirty trick above
        #   - or keep as is and don't waste time
        # session = requests.Session()
        # r = session.post('https://www.cityscapes-dataset.com/login/', data=login)
        # r = session.get('https://www.cityscapes-dataset.com/file-handling/?packageID=1')
        # print(r.request.url)
        #
        # session.close()

        archives = glob.glob("resources/*.zip")

        if not os.path.exists(self.root):
            os.makedirs(self.root)

        for arch in archives:
            print('Unzipping ' + arch)
            with zipfile.ZipFile(arch, 'r') as zip_ref:
                zip_ref.extractall(self.root)
            os.remove(arch)

        return
