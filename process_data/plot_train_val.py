import pandas as pd
import matplotlib.pyplot as plt


#### plots train and test perofrmance on primary classification and uniformity classification

def calculate_accuracy(targets, predictions):
    correct = (targets == predictions).sum()
    total = len(targets)
    return correct / total

def load_data():
    train_clean_data = pd.read_csv('1cyc_15epoch_exp_all_epoch_res0.01/train_data/train_clean_data.csv')
    train_adv_data = pd.read_csv('1cyc_15epoch_exp_all_epoch_res0.01/train_data/train_adv_data.csv')
    train_losses = pd.read_csv('1cyc_15epoch_exp_all_epoch_res0.01/train_data/train_losses.csv')

    val_clean_data = pd.read_csv('1cyc_15epoch_exp_all_epoch_res0.01/val_data/val_clean_data.csv')
    val_adv_data = pd.read_csv('1cyc_15epoch_exp_all_epoch_res0.01/val_data/val_adv_data.csv')
    val_losses = pd.read_csv('1cyc_15epoch_exp_all_epoch_res0.01/val_data/val_losses.csv')

    return train_clean_data, train_adv_data, train_losses, val_clean_data, val_adv_data, val_losses

def plot_training_classification_accuracies(train_clean_data, train_adv_data):
    epochs = sorted(train_clean_data['epoch'].unique())

    train_clean_class_acc = []
    train_adv_class_acc = []
    train_clean_uni_acc = []
    train_adv_uni_acc = []

    for epoch in epochs:
        clean_epoch_data = train_clean_data[train_clean_data['epoch'] == epoch]
        adv_epoch_data = train_adv_data[train_adv_data['epoch'] == epoch]

        train_clean_class_acc.append(calculate_accuracy(clean_epoch_data['class targets'], clean_epoch_data['class predictions']))
        train_adv_class_acc.append(calculate_accuracy(adv_epoch_data['class targets'], adv_epoch_data['class predictions']))

        train_clean_uni_acc.append(calculate_accuracy(clean_epoch_data['uni targets'], clean_epoch_data['uni predictions']))
        train_adv_uni_acc.append(calculate_accuracy(adv_epoch_data['nonuni targets'], adv_epoch_data['nonuni predictions']))

    plt.figure()
    plt.plot(epochs, train_clean_class_acc, label="Train Clean Class Accuracy")
    plt.plot(epochs, train_adv_class_acc, label="Train Adv Class Accuracy")
    plt.plot(epochs, train_clean_uni_acc, label="Train Clean Uniformity Accuracy")
    plt.plot(epochs, train_adv_uni_acc, label="Train Adv Uniformity Accuracy")

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Classification Accuracies Over Epochs')
    plt.legend()
    plt.savefig('training_classification_accuracies.png')
    plt.show()

def plot_training_primary_classification_losses(train_losses):
    epochs = train_losses['epoch']

    plt.figure()
    plt.plot(epochs, train_losses['class loss'], label="Train Class Loss")

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Losses for Primary Feature Classification Task')
    plt.legend()
    plt.savefig('training_primary_classification_losses.png')
    plt.show()

def plot_training_uniformity_losses(train_losses):
    epochs = train_losses['epoch']

    plt.figure()
    plt.plot(epochs, train_losses['uniformity loss'], label="Train Uniformity Loss")

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Losses for Uniformity Classification Task')
    plt.legend()
    plt.savefig('training_uniformity_losses.png')
    plt.show()

def plot_validation_classification_accuracies(val_clean_data, val_adv_data):
    epochs = sorted(val_clean_data['epoch'].unique())

    val_clean_class_acc = []
    val_adv_class_acc = []
    val_clean_uni_acc = []
    val_adv_uni_acc = []

    for epoch in epochs:
        clean_epoch_data = val_clean_data[val_clean_data['epoch'] == epoch]
        adv_epoch_data = val_adv_data[val_adv_data['epoch'] == epoch]

        val_clean_class_acc.append(calculate_accuracy(clean_epoch_data['class targets'], clean_epoch_data['class predictions']))
        val_adv_class_acc.append(calculate_accuracy(adv_epoch_data['class targets'], adv_epoch_data['class predictions']))

        val_clean_uni_acc.append(calculate_accuracy(clean_epoch_data['uni targets'], clean_epoch_data['uni predictions']))
        val_adv_uni_acc.append(calculate_accuracy(adv_epoch_data['nonuni targets'], adv_epoch_data['nonuni predictions']))

    plt.figure()
    plt.plot(epochs, val_clean_class_acc, label="Val Clean Class Accuracy")
    plt.plot(epochs, val_adv_class_acc, label="Val Adv Class Accuracy")
    plt.plot(epochs, val_clean_uni_acc, label="Val Clean Uniformity Accuracy")
    plt.plot(epochs, val_adv_uni_acc, label="Val Adv Uniformity Accuracy")

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Classification Accuracies Over Epochs')
    plt.legend()
    plt.savefig('validation_classification_accuracies.png')
    plt.show()

def plot_validation_primary_classification_losses(val_losses):
    epochs = val_losses['epoch']

    plt.figure()
    plt.plot(epochs, val_losses['clean class loss'], label="Val Clean Class Loss")
    plt.plot(epochs, val_losses['adv class loss'], label="Val Adv Class Loss")

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Losses for Primary Feature Classification Task')
    plt.legend()
    plt.savefig('validation_primary_classification_losses.png')
    plt.show()

def plot_validation_uniformity_losses(val_losses):
    epochs = val_losses['epoch']

    plt.figure()
    plt.plot(epochs, val_losses['uniformity loss'], label="Val Uniformity Loss")

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Losses for Uniformity Classification Task')
    plt.legend()
    plt.savefig('validation_uniformity_losses.png')
    plt.show()

if __name__ == "__main__":
    train_clean_data, train_adv_data, train_losses, val_clean_data, val_adv_data, val_losses = load_data()

    # training accuracies
    plot_training_classification_accuracies(train_clean_data, train_adv_data)

    # training losses (primary classification task)
    plot_training_primary_classification_losses(train_losses)

    # training losses (uniformity task)
    plot_training_uniformity_losses(train_losses)

    # validation accuracies
    plot_validation_classification_accuracies(val_clean_data, val_adv_data)

    # validation losses (primary classification task)
    plot_validation_primary_classification_losses(val_losses)

    # validation losses (uniformity task)
    plot_validation_uniformity_losses(val_losses)
