from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton

app = QApplication([])

window = QWidget()
window.setWindowTitle("PyQt Example")
window.resize(300, 200)

label = QLabel("Hello, PyQt!", parent=window)
label.move(100, 50)

button = QPushButton("Click Me", parent=window)
button.move(150, 100)

window.show()
app.exec()

