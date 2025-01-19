from model import TrackNet3
from datasets import TrackNetDataset
from custom_callback import ValidationCallback
import argparse, os
from pathlib import Path
import tensorflow as tf
from keras import optimizers

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model


if __name__ == '__main__':
    root = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_model_path", type=str, default=os.path.join(root, 'models'))
    parser.add_argument("--n_classes", type=int, default=256)
    parser.add_argument("--input_height", type=int, default=360)
    parser.add_argument("--input_width", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100) # TracknetV3 only use 30
    parser.add_argument("--batch_size", type=int, default=10) # 2 change to ?, the batch_size increasing will cause the processing time increasing
    parser.add_argument("--load_model_status", type=str, default=False)
    parser.add_argument("--steps_per_epoch", type=int, default=200) # 200 change to 300, which means more films(batches) are trained.

    args = parser.parse_args()
    batch_size = args.batch_size
    n_classes = args.n_classes
    input_height = args.input_height
    input_width = args.input_width
    save_model_path = args.save_model_path
    epochs = args.epochs
    load_model_status = args.load_model_status
    steps_per_epoch = args.steps_per_epoch

    optimizer_name = optimizers.Adadelta(learning_rate=1.0)
 
    
    model = TrackNet3(input_height=input_height, input_width=input_width)
      
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=optimizer_name, metrics=['accuracy'])

    train_dataset = TrackNetDataset(input_height, input_width, batch_size)
    # stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    validation_callback = ValidationCallback() # This was created by the author for viewing the training results in figures at media
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs",
                                                          histogram_freq=1,
                                                          profile_batch='20,180')
    model_checkpoint = ModelCheckpoint(f'{save_model_path}/tracknet.keras', save_weights_only=False, # model was saved here
                                      #  save_freq=10000, # using so high value means not saving model, save_freq means the number of batches
                                       save_freq="epoch", # save the model per epoch
                                       monitor="loss",
                                       mode='min',
                                       verbose=1)

    model.fit(train_dataset, epochs=epochs, verbose=1, steps_per_epoch=steps_per_epoch,
              # max_queue_size=10,
              # workers=2,
              # use_multiprocessing=True,
              callbacks=[model_checkpoint,validation_callback]
              )
    
    # model.save(f'{save_model_path}/tracknet.keras')
