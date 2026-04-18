import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(y_test, y_pred, title):

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

    plt.title(title)
    plt.show()
