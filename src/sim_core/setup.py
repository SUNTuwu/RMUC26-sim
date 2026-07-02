from glob import glob
import os

from setuptools import setup


package_name = "sim_core"


setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    package_dir={package_name: "src"},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "pynput", "evdev"],
    zip_safe=True,
    maintainer="somo",
    maintainer_email="sunnycat_158@qq.com",
    description="Core simulation nodes for sentry_sim.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "sentry_sim_node = sim_core.sentry_sim_node:main",
            "keyboard_test = sim_core.keyboard_test:main",
            "chassis = sim_core.chassis:main",
            "nav_feedback_adapter = sim_core.nav_feedback_adapter:main",
        ],
    },
)
