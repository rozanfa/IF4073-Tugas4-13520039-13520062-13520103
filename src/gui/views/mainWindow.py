# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui\mainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainGUI(object):
    def setupUi(self, MainGUI):
        MainGUI.setObjectName("MainGUI")
        MainGUI.setWindowModality(QtCore.Qt.NonModal)
        MainGUI.resize(802, 600)
        self.centralwidget = QtWidgets.QWidget(MainGUI)
        self.centralwidget.setEnabled(True)
        self.centralwidget.setMinimumSize(QtCore.QSize(800, 559))
        self.centralwidget.setAutoFillBackground(True)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.formLayout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setRowWrapPolicy(QtWidgets.QFormLayout.DontWrapRows)
        self.formLayout.setLabelAlignment(QtCore.Qt.AlignCenter)
        self.formLayout.setFormAlignment(QtCore.Qt.AlignCenter)
        self.formLayout.setHorizontalSpacing(30)
        self.formLayout.setObjectName("formLayout")
        self.fileLabel = QtWidgets.QLabel(self.centralwidget)
        self.fileLabel.setObjectName("fileLabel")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.fileLabel)
        self.fileWidget = QtWidgets.QWidget(self.centralwidget)
        self.fileWidget.setObjectName("fileWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.fileWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.filePathTxt = QtWidgets.QLineEdit(self.fileWidget)
        self.filePathTxt.setObjectName("filePathTxt")
        self.horizontalLayout.addWidget(self.filePathTxt)
        self.browseFileBtn = QtWidgets.QPushButton(self.fileWidget)
        self.browseFileBtn.setCheckable(False)
        self.browseFileBtn.setObjectName("browseFileBtn")
        self.horizontalLayout.addWidget(self.browseFileBtn)
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.fileWidget)
        self.modelLabel = QtWidgets.QLabel(self.centralwidget)
        self.modelLabel.setObjectName("modelLabel")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.modelLabel)
        self.modelWidget = QtWidgets.QWidget(self.centralwidget)
        self.modelWidget.setObjectName("modelWidget")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.modelWidget)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.modelWidget)
        self.verticalLayout.addLayout(self.formLayout)
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setObjectName("graphicsView")
        self.verticalLayout.addWidget(self.graphicsView)
        self.mediaCntWidget = QtWidgets.QWidget(self.centralwidget)
        self.mediaCntWidget.setObjectName("mediaCntWidget")
        self.mediaContainer = QtWidgets.QVBoxLayout(self.mediaCntWidget)
        self.mediaContainer.setObjectName("mediaContainer")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSpacing(20)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.seekSlider = QtWidgets.QSlider(self.mediaCntWidget)
        self.seekSlider.setMaximum(5)
        self.seekSlider.setOrientation(QtCore.Qt.Horizontal)
        self.seekSlider.setInvertedControls(False)
        self.seekSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.seekSlider.setTickInterval(0)
        self.seekSlider.setObjectName("seekSlider")
        self.horizontalLayout_2.addWidget(self.seekSlider)
        self.timeLbl = QtWidgets.QLabel(self.mediaCntWidget)
        self.timeLbl.setObjectName("timeLbl")
        self.horizontalLayout_2.addWidget(self.timeLbl)
        self.mediaContainer.addLayout(self.horizontalLayout_2)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.playBtn = QtWidgets.QPushButton(self.mediaCntWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.playBtn.sizePolicy().hasHeightForWidth())
        self.playBtn.setSizePolicy(sizePolicy)
        self.playBtn.setObjectName("playBtn")
        self.gridLayout_2.addWidget(self.playBtn, 0, 0, 1, 1)
        self.stopBtn = QtWidgets.QPushButton(self.mediaCntWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.stopBtn.sizePolicy().hasHeightForWidth())
        self.stopBtn.setSizePolicy(sizePolicy)
        self.stopBtn.setObjectName("stopBtn")
        self.gridLayout_2.addWidget(self.stopBtn, 0, 1, 1, 1)
        self.mediaContainer.addLayout(self.gridLayout_2)
        self.verticalLayout.addWidget(self.mediaCntWidget)
        self.predictWidget = QtWidgets.QWidget(self.centralwidget)
        self.predictWidget.setObjectName("predictWidget")
        self.predictContainer = QtWidgets.QHBoxLayout(self.predictWidget)
        self.predictContainer.setObjectName("predictContainer")
        self.predictLabel = QtWidgets.QLabel(self.predictWidget)
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.predictLabel.setFont(font)
        self.predictLabel.setObjectName("predictLabel")
        self.predictContainer.addWidget(self.predictLabel)
        self.predictBtn = QtWidgets.QPushButton(self.predictWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predictBtn.sizePolicy().hasHeightForWidth())
        self.predictBtn.setSizePolicy(sizePolicy)
        self.predictBtn.setObjectName("predictBtn")
        self.predictContainer.addWidget(self.predictBtn)
        self.verticalLayout.addWidget(self.predictWidget)
        MainGUI.setCentralWidget(self.centralwidget)
        self.actionLoad = QtWidgets.QAction(MainGUI)
        self.actionLoad.setObjectName("actionLoad")
        self.actionCNN = QtWidgets.QAction(MainGUI)
        self.actionCNN.setObjectName("actionCNN")
        self.actionNon_CNN = QtWidgets.QAction(MainGUI)
        self.actionNon_CNN.setObjectName("actionNon_CNN")
        self.actionLoad_Video = QtWidgets.QAction(MainGUI)
        self.actionLoad_Video.setObjectName("actionLoad_Video")

        self.retranslateUi(MainGUI)
        QtCore.QMetaObject.connectSlotsByName(MainGUI)

    def retranslateUi(self, MainGUI):
        _translate = QtCore.QCoreApplication.translate
        MainGUI.setWindowTitle(_translate("MainGUI", "Vehicle Detection GUI"))
        self.fileLabel.setText(_translate("MainGUI", "File"))
        self.browseFileBtn.setText(_translate("MainGUI", "Browse File"))
        self.modelLabel.setText(_translate("MainGUI", "Model"))
        self.timeLbl.setText(_translate("MainGUI", "00:00/00:12"))
        self.playBtn.setText(_translate("MainGUI", "Play"))
        self.stopBtn.setText(_translate("MainGUI", "Stop"))
        self.predictLabel.setText(_translate("MainGUI", "Select File to Predict"))
        self.predictBtn.setText(_translate("MainGUI", "Predict"))
        self.actionLoad.setText(_translate("MainGUI", "Load Image"))
        self.actionCNN.setText(_translate("MainGUI", "CNN"))
        self.actionNon_CNN.setText(_translate("MainGUI", "Non-CNN"))
        self.actionLoad_Video.setText(_translate("MainGUI", "Load Video"))
