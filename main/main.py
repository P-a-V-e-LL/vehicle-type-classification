import sys
from PyQt5 import QtWidgets
from main_Window import *
from add_window import *
#from config_actions import *
#from vtc.classifier import *
def main():
    #data = config_read()
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    #ui.statusbar_text()
    MainWindow.show()
    sys.exit(app.exec_()+ui.close_db_connection())

def add_window():
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_db_add_window()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    #add_window()
    main()
