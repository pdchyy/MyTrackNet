
from tensorflow import keras
from accuracy import validate
import cv2,csv
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from glob import glob
from collections import defaultdict
from pathlib import Path

class ValidationCallback(keras.callbacks.Callback):

    def on_train_begin(self, logs=None):
        self.losses = []
        self.epochs = []
        self.f1 = []
        self.precision = []
        self.recall = []
        self.metrics_epochs = []
        self.val_loss = []
        validation_data = defaultdict(list)
        root = Path(__file__).parent
        validation_csv_path = os.path.join(root, 'labels_val.csv')

        with open(validation_csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row
            for row in reader:
                for i, v in enumerate(row):
                    validation_data[i].append(v)
        self.validation_data = validation_data
        print('Beginning training......')

    def on_epoch_begin(self, epoch, logs=None):
        if (epoch+1) % 100 == 0:
            print("Validating..... ")
            val_loss, f1, precision, recall = validate(self.model, self.validation_data)
            self.metrics_epochs.append(epoch + 1)
            self.f1.append(f1)
            self.precision.append(precision)
            self.recall.append(recall)
            self.val_loss.append(val_loss)


    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch + 1)
        self.losses.append(logs['loss'])


    def on_train_end(self, logs=None):
        print("Stop training.....")

        # Plot and save the Training Loss
        plt.figure(figsize=(8, 6))
        plt.plot(self.epochs, self.losses,label='Training Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.legend()
        plt.savefig('media/training_loss_plot.png')
        plt.close()
        
        # Plot and save the F1 score
        plt.figure(figsize=(8, 6))
        plt.plot(self.metrics_epochs, self.f1, label='f1 score', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('f1')
        plt.legend()
        plt.savefig('media/validation_f1_score_plot.png')
        plt.close()

        # Plot and save the Precision
        plt.figure(figsize=(8, 6))
        plt.plot(self.metrics_epochs, self.precision, label='Precision', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()
        plt.savefig('media/validation_precision_score_plot.png')
        plt.close()

        # Plot and save the Recall
        plt.figure(figsize=(8, 6))
        plt.plot(self.metrics_epochs, self.recall, label='Recall', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()
        plt.savefig('media/validation_recall_score_plot.png')
        plt.close()

        # Plot and save the Validation Loss
        plt.figure(figsize=(8, 6))
        plt.plot(self.metrics_epochs, self.val_loss, label='Val Loss', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.legend()
        plt.savefig('media/val_loss_plot.png')
        plt.close()

