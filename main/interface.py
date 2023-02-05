import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow
from PyQt5.QtGui import QIcon


class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()


    def initUI(self):

        self.setGeometry(1300, 1300, 1300, 1220)
        self.setWindowTitle('Icon')
        self.setWindowIcon(QIcon('web.png'))

        self.show()


class Interface(QWidget):
    pass

def main():
    app = QApplication(sys.argv)
    main = QMainWindow()
    main.setWindowTitle("New Progmain")
    main.setGeometry(300, 300, 400, 400)
    main.show()
    #ex = Example()
    sys.exit(app.exec_())

def config_write(db_path):
    import configparser
    filename = "database.ini"
    config = configparser.ConfigParser()
    config.read(filename)
    config.add_section("SETTINGS")
    config.set("SETTINGS", "start_db", db_path)
    with open(filename, "w") as config_file:
        config.write(config_file)

def config_read():
    import configparser
    filename = "database.ini"
    config = configparser.ConfigParser()
    config.read(filename)
    return config.get("SETTINGS", "start_db"))

if __name__ == '__main__':
    config_read()
