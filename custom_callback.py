
from tensorflow import keras
from accuracy import validate, validate_1
import cv2,csv
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from glob import glob
from collections import defaultdict
from pathlib import Path
import time

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
        self.validation_data = validation_data # validation_data is a dictionary
        print('Beginning training......')

    def on_epoch_begin(self, epoch, logs=None):
        # if (epoch+1) % 100 == 0: # record the value per 100 epochs
        if (epoch+1) % 50 == 0: # must use epoch+1 since originally the first eopch is 0.
        
            print(f"Validating..... at epoch={epoch+1}")
            start = time.time()
            # val_loss, f1, precision, recall = validate(self.model, self.validation_data) # The author created validate() from accuracy.py for SCCE loss
            val_loss, f1, precision, recall = validate_1(self.model, self.validation_data) # The author created validate() from accuracy.py for WBCE_loss
            self.metrics_epochs.append(epoch + 1) # since (epoch+1) % 50 == 0
            self.f1.append(f1)
            self.precision.append(precision)
            self.recall.append(recall)
            self.val_loss.append(val_loss)
            print(f"Validation time: {time.time()-start}")
            print("Validation results:")
            print('precision = {}'.format(self.precision[-1]))
            print('recall = {}'.format(self.recall[-1]))
            print('f1 = {}'.format(self.f1[-1]))
            print("Validation loss:",self.val_loss[-1])
            


    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch + 1) # Since "if (epoch+1) % 50 == 0:"  in on_epoch_begin()
        self.losses.append(logs['loss'])
        print("On epoch end")
        

    def on_train_end(self, logs=None):
        print("Stop training....., the final:")
        print('precision = {}'.format(self.precision[-1]))
        print('recall = {}'.format(self.recall[-1]))
        print('f1 = {}'.format(self.f1[-1]))
        print("Validation loss:",self.val_loss[-1])
        

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

