from pathlib import Path
import sys

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QRadioButton, QButtonGroup,
    QGraphicsScene
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from non_cnn.model import FFTModel, get_image
from gui.views.nonCnnWindow import Ui_NonCNNPredict


MODELS = {
    'SVC': 'modelSVC4.pkl',
}


class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_NonCNNPredict()
        self.ui.setupUi(self)
        self.connectSignalsSlots()
        self.setupModel()

    def setupModel(self):
        self.current_model = list(MODELS.keys())[0]
        self.models = {
            k: FFTModel.load(Path(__file__).parent / f'../../non_cnn/models/{m}')
            for k, m in MODELS.items()
        }
        btn_group = QButtonGroup(self.ui.modelWidget)
        mdl_layout = self.ui.modelWidget.layout()
        for k in self.models.keys():
            btn = QRadioButton(k, self.ui.modelWidget)
            btn_group.addButton(btn)
            btn.clicked.connect(lambda: self.changeModel(k))
            mdl_layout.addWidget(btn)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)

        scene = self.ui.graphicsView.scene()
        if scene:
            self.ui.graphicsView.fitInView(
                scene.sceneRect(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )

    def changeModel(self, model_name):
        self.current_model = model_name

    def connectSignalsSlots(self):
        self.ui.browseFileBtn.clicked.connect(self.browseFile)
        self.ui.predictBtn.clicked.connect(self.predict)
        self.ui.graphicsView.receivers

    def browseFile(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Image")
        if file_path:
            scene = QGraphicsScene(self.ui.graphicsView)
            scene.addPixmap(QPixmap(file_path))
            self.ui.graphicsView.setScene(scene)
            self.ui.graphicsView.fitInView(scene.sceneRect(),
                                           Qt.AspectRatioMode.KeepAspectRatio)
            self.ui.predictLabel.setText('Click Predict to get prediction')
            self.ui.filePathTxt.setText(file_path)

    def predict(self):
        file_path = self.ui.filePathTxt.text()
        if not file_path:
            return
        model = self.models[self.current_model]
        predictions = model.single_predict(get_image(file_path))
        predictions = dict(sorted(
            predictions.items(), key=lambda x: x[1], reverse=True
        ))
        predictions = ', '.join([
            f'{k} ({v*100:.2f}%)' for k, v in predictions.items()
        ])
        self.ui.predictLabel.setText(
            f'Predictions: {predictions}'
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec())
