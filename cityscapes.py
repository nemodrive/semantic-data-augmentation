import os
import subprocess
import glob
import zipfile
from torchvision import datasets


class Cityscapes:
    CityscapesPackages = {
        'gtFine': '1',
        'gtCoarse': '2',
        'leftImg8bit': '3',
        'leftImg8bit-extra': '4',
        'camera': '8',
        'camera-extra': '9',
        'vehicle': '10',
        'vehicle-extra': '11',
        'leftImg8bit-demo': '12',
        'gtBboxCityPersons': '28'
    }

    CityscapesClass = datasets.Cityscapes.CityscapesClass
    classes = datasets.Cityscapes.classes
    def __init__(self, root, login, packages=('gtFine', 'gtCoarse', 'leftImg8bit'), split='train', mode='fine',
                 target_type='instance', transform=None, target_transform=None, transforms=None, download=False):

        self.root = root

        # Additional wrapper parameters
        self.login = login
        self.packages = packages

        if download:
            for p in packages:
                if not self.__check_pkg_exists(p):
                    self.__download(p)
            self.__unpack_packages()

        self.obj_wrapper = datasets.Cityscapes(root, split, mode, target_type,
                                               transform, target_transform, transforms)

        # Pytorch class parameters
        self.mode = mode
        self.target_type = target_type
        self.split = split
        self.images_dir = self.obj_wrapper.images_dir
        self.targets_dir = self.obj_wrapper.targets_dir
        self.images = self.obj_wrapper.images
        self.targets = self.obj_wrapper

    def __check_pkg_exists(self, package):
        pkg_path = os.path.join(self.root, package)
        return os.path.exists(pkg_path) and len(os.listdir(pkg_path)) != 0

    def __download(self, package):
        # TODO: See whats happening with these requests and delete the dirty trick above
        #   - or keep as is and don't waste time
        # session = requests.Session()
        # r = session.post('https://www.cityscapes-dataset.com/login/', data=login)
        # r = session.get('https://www.cityscapes-dataset.com/file-handling/?packageID=1')
        # print(r.request.url)
        # session.close()

        subprocess.check_call([
            'resources/cityscapes-download-script.sh',
            self.login[0],
            self.login[1],
            self.CityscapesPackages[package]
        ])

    def __unpack_packages(self):
        archives = glob.glob("resources/*.zip")
        if not archives:
            return

        if not os.path.exists(self.root):
            os.makedirs(self.root)

        for arch in archives:
            with zipfile.ZipFile(arch, 'r') as zip_ref:
                zip_ref.extractall(self.root)
            os.remove(arch)

    def __getitem__(self, index):
        return self.obj_wrapper.__getitem__(index)

    def __len__(self):
        return self.obj_wrapper.__len__()

    def extra_repr(self):
        return self.obj_wrapper.extra_repr()

    def _load_json(self, path):
        return self.obj_wrapper._load_json(path)

    def _get_target_suffix(self, mode, target_type):
        return self.obj_wrapper._get_target_suffix(mode, target_type)
