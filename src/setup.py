from setuptools import setup, find_packages

setup(
    name='vehiclass',
    fullname='Pengcit\'s Vehicle Classification',
    version='0.1',
    packages=find_packages(),
    description='Package untuk klasifikasi kendaraan',
    install_requires=[
        'scikit-learn',
        'numpy',
        'pandas',
        'Pillow',
        'matplotlib',
        'pyqt5',
        'ultralytics',
        'opencv-python'
    ],
)
